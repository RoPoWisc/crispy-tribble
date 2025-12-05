import asyncio
import json
import os
import time
import uuid
from typing import List, Dict, Any, Optional

from google import genai
from google.cloud import storage

class GeminiProcessor:
    def __init__(
        self,
        key_file_path: str,
        project_id: str,
        location: str = "global",
        # Defaulting to Gemini 3.0 Pro Preview (Feb 2025 version)
        model_name: str = "gemini-2.5-flash", 
        staging_bucket: str = None,
        system_instruction: str = None
    ):
        """
        Unified Gemini Processor for both Real-Time Async and Batch Jobs.
        
        Args:
            key_file_path: Path to service account JSON key.
            project_id: GCP Project ID.
            location: Vertex AI Region (Gemini 3 is often in global).
            model_name: The model ID (e.g., gemini-3-pro-preview).
            staging_bucket: GCS bucket name (REQUIRED for Batch Jobs).
            system_instruction: Optional system prompt for all requests.
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.staging_bucket = staging_bucket
        self.system_instruction = system_instruction
        self.queue = []
        self._job_registry: Dict[str, Dict[str, Any]] = {}

        # 1. Setup Authentication
        if not os.path.exists(key_file_path):
            raise FileNotFoundError(f"Key file not found: {key_file_path}")
            
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file_path

        # 2. Initialize the New GenAI Client (Vertex Mode)
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )

        # 3. Initialize Storage Client (for Batch Jobs)
        self.storage_client = storage.Client.from_service_account_json(key_file_path)

    # ==========================================
    #  MODE A: Real-Time Async (Fast, Higher Cost)
    # ==========================================

    def add_realtime_request(self, prompt: str, id: str = None, config: dict = None):
        """Add a request to the real-time processing queue."""
        self.queue.append({
            "id": id or str(len(self.queue)),
            "prompt": prompt,
            "config": config
        })

    async def _process_single(self, item: Dict, semaphore: asyncio.Semaphore):
        """Worker for real-time requests."""
        async with semaphore:
            try:
                # Merge default config with item config
                run_config = {}
                if item["config"]:
                    run_config.update(item["config"])
                
                # Add system instruction if present
                if self.system_instruction:
                    run_config['system_instruction'] = self.system_instruction

                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=item["prompt"],
                    config=run_config
                )
                return {"id": item["id"], "status": "success", "output": response.text}
            except Exception as e:
                return {"id": item["id"], "status": "error", "error": str(e)}

    async def run_async_batch(self, max_concurrency: int = 5):
        """Execute the queued requests immediately in parallel."""
        if not self.queue:
            return []
            
        print(f"🚀 [Real-Time] Dispatching {len(self.queue)} requests...")
        sem = asyncio.Semaphore(max_concurrency)
        tasks = [self._process_single(item, sem) for item in self.queue]
        
        results = await asyncio.gather(*tasks)
        self.queue = [] # Clear queue after run
        return results

    # ==========================================
    #  MODE B: Batch Prediction Job (Slow, 50% Cost)
    # ==========================================

    def _normalize_batch_request(self, payload: Any) -> Dict[str, Any]:
        """Convert a user payload into the JSONL shape Vertex batch expects."""
        if isinstance(payload, str):
            request_body = {
                "contents": [
                    {"role": "user", "parts": [{"text": payload}]}
                ]
            }
        elif isinstance(payload, dict):
            if "request" in payload:
                return payload
            # Allow passing just the inner request object.
            request_body = payload
        else:
            raise TypeError("Each payload must be either a prompt string or a request dict.")

        if self.system_instruction and "system_instruction" not in request_body:
            request_body["system_instruction"] = {"parts": [{"text": self.system_instruction}]}

        return {"request": request_body}

    def submit_batch_job(
        self,
        model_name: Optional[str],
        requests: List[Any],
        job_display_name: Optional[str] = None,
    ) -> str:
        """Create a Vertex Gemini batch job and return the job resource name."""
        if not self.staging_bucket:
            raise ValueError("staging_bucket must be set in __init__ to use batch jobs.")
        if not requests:
            raise ValueError("At least one request is required to start a batch job.")

        safe_label = (job_display_name or f"batch-{uuid.uuid4()}").replace(" ", "_")
        input_uri = f"gs://{self.staging_bucket}/inputs/{safe_label}.jsonl"
        output_prefix = f"gs://{self.staging_bucket}/outputs/{safe_label}"

        serialized = [self._normalize_batch_request(r) for r in requests]
        bucket = self.storage_client.bucket(self.staging_bucket)
        bucket.blob(f"inputs/{safe_label}.jsonl").upload_from_string(
            "\n".join(json.dumps(line) for line in serialized)
        )
        print(f"📦 [Batch Job] Uploaded {len(serialized)} requests to {input_uri}")

        batch_job = self.client.batches.create(
            model=model_name or self.model_name,
            src=input_uri,
            config={"dest": output_prefix},
        )
        print(f"   Submitted job {batch_job.name} targeting {output_prefix}")

        self._job_registry[batch_job.name] = {
            "bucket": self.staging_bucket,
            "safe_label": safe_label,
            "requests": requests,
        }
        return batch_job.name

    def check_job_status(self, job_name: str) -> str:
        """Return the Vertex job state string, e.g., JOB_STATE_SUCCEEDED."""
        job_status = self.client.batches.get(name=job_name)
        return str(job_status.state)

    def download_results(self, job_name: str) -> List[Dict[str, Any]]:
        """Download prediction files for a finished job and return parsed responses."""
        meta = self._job_registry.get(job_name)
        if not meta:
            raise KeyError(f"Job metadata not found for {job_name}")

        bucket = self.storage_client.bucket(meta["bucket"])
        safe_label = meta["safe_label"]
        requests_payload = meta.get("requests", [])

        responses: List[Dict[str, Any]] = []
        blobs = list(bucket.list_blobs(prefix=f"outputs/{safe_label}"))
        idx = 0
        for blob in blobs:
            if not (blob.name.endswith(".jsonl") and "predictions" in blob.name):
                continue
            content = blob.download_as_text()
            for line in content.splitlines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    responses.append(
                        {
                            "index": idx,
                            "request": requests_payload[idx] if idx < len(requests_payload) else None,
                            "response_text": f"[Error Parsing Result: {exc}]",
                            "raw": line,
                        }
                    )
                    idx += 1
                    continue

                candidates = data.get("response", {}).get("candidates", [])
                text = ""
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                responses.append(
                    {
                        "index": idx,
                        "request": requests_payload[idx] if idx < len(requests_payload) else None,
                        "response_text": text or "[No Content Generated]",
                        "raw": data,
                    }
                )
                idx += 1

        print(f"   Downloaded {len(responses)} responses for {job_name}")
        return responses

    def run_batch_job(self, prompts: List[str], poll_interval: int = 30) -> List[str]:
        """Submit prompts as a batch job, poll until completion, then return response text."""
        print(f"📦 [Batch Job] Starting process for {len(prompts)} prompts...")
        job_name = self.submit_batch_job(model_name=self.model_name, requests=prompts)
        print("   Waiting for completion (this may take minutes/hours)...")

        while True:
            state = self.check_job_status(job_name)
            if "SUCCEEDED" in state:
                print("\n   ✅ Job Succeeded!")
                break
            if "FAILED" in state or "CANCELLED" in state:
                raise RuntimeError(f"Batch Job Failed: {state}")
            print(".", end="", flush=True)
            time.sleep(poll_interval)

        result_payloads = self.download_results(job_name)
        return [item.get("response_text", "") for item in result_payloads]

    def generate(self, system: str, user: str) -> str:
        """Generate content using the real-time API."""
        try:
            contents = []
            if system:
                contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "user", "parts": [{"text": user}]})

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            return response.text
        except Exception as e:
            # Enhanced debug logging for Gemini failures
            error_type = type(e).__name__
            user_preview = (user[:200] + "...") if user and len(user) > 200 else (user or "")
            system_len = len(system) if system else 0
            user_len = len(user) if user else 0

            print("Gemini request failed:")
            print(f"  error_type     = {error_type}")
            print(f"  project_id     = {self.project_id}")
            print(f"  location       = {self.location}")
            print(f"  model_name     = {self.model_name}")
            print(f"  system_len     = {system_len}")
            print(f"  user_len       = {user_len}")
            print(f"  user_preview   = {user_preview!r}")
            print(f"  raw_exception  = {e!r}")

            # Optionally, if you want a full traceback in logs:
            # import traceback
            # traceback.print_exc()

            return ""

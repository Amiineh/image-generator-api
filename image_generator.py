import asyncio
import logging
import fal
from pydantic import BaseModel, Field
from typing import Optional, Literal
from fal.toolkit.image import ImageSize, ImageSizeInput
from datetime import datetime
from enum import Enum
from uuid import uuid4

class Input(BaseModel):
    """Input model for text-to-image & image-to-iamge generation."""
    prompt: str = Field(
        description="The text prompt to guide image generation.",
        examples=["A beautiful landscape painting of a park with river and flowers"]
    )
    negative_prompt: str = Field(
        description="Negative prompt for image generation.",
        default=""
    )
    seed: Optional[int] = Field(
        description="Random seed for reproducibility. If None, a random seed is chosen.",
        default=None
    )
    num_inference_steps: int = Field(
        description="Number of inference steps for sampling. Higher values give better quality but take longer.",
        default=27
    )
    enable_safety_checker: bool = Field(
        description="If set to true, input data will be checked for safety before processing.",
        default=False
    )
    enable_output_safety_checker: bool = Field(
        description="If set to true, output image will be checked for safety after generation.",
        default=False
    )
    enable_prompt_expansion: bool = Field(
        description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning.",
        default=False
    )
    acceleration: Literal["regular", "none"] = Field(
        description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.",
        default="regular"
    )
    guidance_scale: float = Field(
        description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality",
        default=3.5
    )
    guidance_scale_2: float = Field(
        description="Guidance scale for the second stage of the model. This is used to control the adherence to the prompt in the second stage of the model",
        default=4
    )
    shift: float = Field(
        description="Shift value for the image. Must be between 1.0 and 10.0.",
        default=2.0,
        ge=1.0,
        le=10.0
    )
    image_size: ImageSizeInput = Field(
        description="The size of the generated image.",
        default=ImageSize(width=512, height=512),
    )
    image_format: Literal["jpeg", "png"] = Field(
        description="The format of the output image.",
        default="jpeg"
    )
    aspect_ratio: Literal[ "16:9", "9:16", "1:1"] = Field(
        description="Aspect ratio of the generated image.",
        default="1:1"
    )
    image_url: str | None = Field(
        description="URL of the input image. Required for image-to-image generation.",
        default=None
    )

class Output(BaseModel):
    """Output model for image generation."""
    image: str = Field(
        description="The generated image file."
    )
    prompt: str = Field(
        description="The text prompt used for image generation."
    )
    seed: int = Field(
        description="The seed used for generation."
    )


class JobStatus(Enum):
    SCHEDULED="scheduled",
    RUNNING="running",
    COMPLETED="completed",
    STOPPED="stopped",
    FAILED="failed"


# in a real-world scenario like production, this could be saved in a DB
class Job:
    """ each request is saved as a job and stored in a queue for processing and retries """
    capability: str
    params: Input
    attempt: int=0
    created: datetime
    last_run: datetime = None
    status: JobStatus
    errors: list
    result: Output

    def __init__(self, capability, params):
        self.job_id = str(uuid4())
        self.capability = capability
        self.params = params
        self.created = datetime.now()

    def get_status(self):
        return self.status or 'Unknown'


class Worker:
    """ This class handles jobs, by storing them in a dict (in real world this would be DB), and keeping job_ids in a queue"""
    job_id_queue = asyncio.Queue()
    jobs = {} # for keeping track of jobs. In production this could be DB

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def __init__(self, max_try=4, wait_times:list[int]=None):
        self.max_try = max_try
        self.wait_times=[1, 5, 30, 60] # wait this many seconds between consecutive tries
        if wait_times and len(wait_times) != self.max_try:
            raise ValueError(f"wait_times should have ${self.max_try} values")
        self.created = datetime.now()


    async def add_job(self, capability, params):
        try:
            job = Job(capability, params)
            job_id = job.job_id
            await self.job_id_queue.put(job_id)
            self.jobs[job_id] = job
            job.status = JobStatus.SCHEDULED
            self.logger.info(f"Added job {job_id} to queue")
            return job_id
        except Exception as e:
            self.logger.info(f"Error adding job: {e}")

    def stop_job(self, job_id):
        if job_id not in self.jobs.keys():
            raise ValueError("job_id is not valid or is no longer in memory")
        self.jobs[job_id].status = JobStatus.STOPPED
        self.logger.info(f"Stopped job {job_id}")
        return

    async def retry_job(self, job_id):
        if job_id not in self.jobs.keys():
            raise ValueError("job_id is not valid or is no longer in memory")
        await self.job_id_queue.put(job_id)
        self.logger.info(f"Retrying job {job_id}")
        return

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def update_job(self, job_id, new_job):
        if job_id not in self.jobs.keys():
            raise ValueError("job_id is not valid or is no longer in memory")
        self.jobs[job_id] = new_job
        self.logger.info(f"Updated job with id {job_id} to {new_job}")

class WanImageGenerator(fal.App):
    """ The main fal serverless app for Wan image generation """
    wan_endpoints = {
        "t2i": "fal-ai/wan/v2.2-a14b/text-to-image",
        "i2i": "fal-ai/wan/v2.2-a14b/image-to-image"
    }
    keep_alive = 300
    app_name = "image_generator"
    machine_type = "GPU-H100"
    requirements = [
        "fal_client",
        "pillow",
        "requests",
    ]

    def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.worker = Worker()
        self.consumer_task = asyncio.create_task(self.queue_consumer())
        self.logger.info("Wan 2.2. image generator initialized!")

    async def queue_consumer(self):
        """ Retrieves jobs from queue and executes them, Will retry on fail for max_try times """
        # in a prod this could be an sqs consumer
        self.logger.info("Queue consumption started")
        while True:
            try:
                try:
                    job_id = await asyncio.wait_for(self.worker.job_id_queue.get(), timeout=1)
                except asyncio.TimeoutError:
                    # no job available
                    continue

                job = self.worker.get_job(job_id)
                if not job:
                    self.logger.info(f"No job found with id {job_id}")
                    continue

                if job.status not in [JobStatus.SCHEDULED, JobStatus.FAILED]:
                    self.logger.info(f"Job {job.job_id} status is {job.status}. Moving on.")
                    continue

                # if retrying jobs, wait before execution in case api connection was down or sth
                if job.last_run is not None and job.attempt > 0:
                    time_lapsed = (datetime.now() - job.last_run).total_seconds()
                    wait_time = self.worker.wait_times[job.attempt - 1]
                    if (time_lapsed < wait_time):
                        remaining_time = wait_time - time_lapsed
                        self.logger.info(f"Waiting {remaining_time} seconds before running job {job_id} again")
                        await asyncio.sleep(remaining_time)

                await self.execute_job(job_id)
            except Exception as e:
                self.logger.info(f"Error consuming the job queue: {e}")
                await asyncio.sleep(1)

    async def execute_job(self, job_id):
        """ runs the job based on capability (t2i or i2i) """
        try:
            job = self.worker.get_job(job_id)
            self.logger.info(f"Executing job {job_id} with capability {job.capability} - attempt {job.attempt}")
            job.attempt += 1
            job.status = JobStatus.RUNNING
            job.last_run = datetime.now()

            if job.capability == 't2i':
                result = await self.text_to_image(job.params)
            elif job.capability == 'i2i':
                result = await self.image_to_image(job.params)
            else:
                raise ValueError(f"Unknown capability {job.capability}")

            job.result = result
            job.status = JobStatus.COMPLETED
            self.logger.info(f"Job {job_id} executed successfully! result: {result}")
            self.worker.update_job(job_id, new_job=job)
            return result
        except Exception as e:
            job = self.worker.get_job(job_id)
            self.logger.info(f"Attempt {job.attempt} for job {job_id} failed with error {e}")
            job.errors.append(e)
            if job.attempt < self.worker.max_try:
                # retry job
                job.status = JobStatus.SCHEDULED
                self.worker.update_job(job_id, new_job=job)
                self.logger.info(f"Retrying job {job_id}")
                await self.worker.retry_job(job_id)
            else:
                self.logger.info(f"Job {job_id} failed {self.worker.max_try} times. No longer trying.")
                job.status = JobStatus.FAILED
                self.worker.update_job(job_id, new_job=job)

    def download_image_and_upload_to_fal(self, url):
        # To make sure we're not exposing urls
        import fal_client
        from PIL import Image
        import requests
        from io import BytesIO

        try:
            self.logger.info(f"Downloading image from {url}")
            response = requests.get(url)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            uploaded_url = fal_client.upload_image(img)
            self.logger.info(f"Uploaded image to {uploaded_url}")
            return uploaded_url
        except Exception as e:
            self.logger.info(f"Error while downloading/uploading image: {str(e)}")
            raise e

    @fal.endpoint("/")
    def home(self) -> dict:
        return {
            "success": True,
            "message": "Welcome to Wan 2.2 image generator!",
            "endpoints": {
                "/": "general info",
                "t2i": "text-to-image generation -> returns job_id",
                "i2i": "image-to-image generation -> returns job_id",
                "/job/status/{job_id}": "get status of the requested job",
                "job_queue/status": "get status of all of the jobs in queue"
            }
        }

    @fal.endpoint("/t2i")
    async def handle_t2i_req(self, req: Input) -> dict:
        try:
            self.logger.info("received t2i request")
            job_id = await self.worker.add_job(capability='t2i', params=req)
            self.logger.info(f"Queued job with id {job_id}")
            return {
                "success": True,
                "message": "Job queued successfully",
                "data": {"job_id": job_id}
            }
        except Exception as e:
            self.logger.info(f"Error queueing t2i job: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to queue request"
            }

    async def text_to_image(self, req: Input) -> Output:
        import fal_client

        handler = await fal_client.submit_async(
            self.wan_endpoints["t2i"],
            arguments=req.model_dump(mode='json', exclude_none=True, by_alias=True)
        )
        async for event in handler.iter_events(with_logs=True):
            self.logger.info(event)
        result = await handler.get()
        self.logger.info(result)

        uploaded_url = self.download_image_and_upload_to_fal(result["image"]["url"])
        return Output(image=uploaded_url, prompt=req.prompt, seed=result["seed"])

    @fal.endpoint("/i2i")
    async def handle_i2i_req(self, req: Input):
        try:
            if not req.image_url or not req.image_url.strip():
                return {
                    "success": False,
                    "error": "image_url is required for image-to-image generation",
                    "message": "missing parameter image_url"
                }
            self.logger.info("received i2i request")
            job_id = await self.worker.add_job(capability='i2i', params=req)
            self.logger.info(f"Queued job with id {job_id}")
            return {
                "success": True,
                "message": "Job queued successfully",
                "data": {"job_id": job_id}
            }
        except Exception as e:
            self.logger.info(f"Error queuing job for i2i: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to queue request",
            }

    async def image_to_image(self, req: Input):
        import fal_client

        if not req.image_url or not req.image_url.strip():
            raise ValueError("image_url cannot be empty")

        # replace input image url with uploaded url before calling endpoint
        uploaded_url = self.download_image_and_upload_to_fal(req.image_url)
        arguments = req.model_dump(mode='json', exclude_none=True, by_alias=True)
        arguments["image_url"] = uploaded_url
        handler = await fal_client.submit_async(
            self.wan_endpoints['i2i'],
            arguments=arguments
        )
        request_id = handler.request_id
        self.logger.info(f"request_id: {request_id}")
        async for event in handler.iter_events(with_logs=True):
            self.logger.info(event)

        result = await handler.get()
        self.logger.info(result)

        uploaded_url = self.download_image_and_upload_to_fal(result["image"]["url"])
        return Output(image=uploaded_url, prompt=req.prompt, seed=result["seed"])

    @fal.endpoint("jobs/status/{job_id}")
    def get_job_status(self, job_id):
        try:
            job = self.worker.get_job(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            result = {
                "success": True,
                "data": {
                    "job_id": job_id,
                    "status": job.status,
                    "created": job.created,
                    "attempt": job.attempt,
                    "max_try": self.worker.max_try
                }
            }
            self.logger.info(f"Job status: {result}")
            return result
        except Exception as e:
            self.logger.info(f"Error getting job ({job_id}) status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get status for job {job_id}"
            }

    @fal.endpoint("/job_queue/status")
    def get_queue_status(self) -> dict:
        try:
            job_data = [{
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "capability": job.capability,
                    "attempt": job.attempt,
                    "created": job.created.isoformat()
                } for job in self.worker.jobs.values()]
            self.logger.info(f"returning job statuses: {job_data}")

            return {
                "success": True,
                "data": {"jobs": job_data}
            }
        except Exception as e:
            self.logger.info(f"Error getting queue status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get queue status"
            }
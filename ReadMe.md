# fal SDK api - Wan 2.2. image generator 

This assessment implements a serverless fal application to call Wan 2.2 text-to-image and image-to-image generation as part of the MLE interview assessment.

## Endpoints
The endpoints of this app are:
- `"/"`: the root which lists the endpoint with a breif explanation
- `"/t2i"`: text-to-image generation, retuns job_id that can be used for status check later
- `"i2i"`: image-to-image generation, also returns job_id
- `"/jobs/status/{job_id}"`: get the status of a job
- `"/job_queue/status"`: returns information about all jobs in the queue

## Retry Mechanism
For reliability, in case the api server is down for different times, an execution queue is implemented.
Each incoming request is saved as a job in the queue with input parameters and capability (t2i or i2i). 
Jobs also have status field, input parameters, output result, created and run time and error stacks.

An asynchronous queue consumer takes the jobs from the queue and executes them. Jobs are retried a maximum of 4 times (adjustable in params), 
and each time the wait time is doubled to give api more time to revive. Wait times are [1, 5, 30, 60] seconds and can be adjusted as class arg in instantiation.

The endpoints return results immediately with job_id that can be used later for status check. 
The queue consumer runs in the background and executes and retries jobs.

## Inputs and Outputs
All of the input and output fields of the original fal Wan 2.2 are implemented as in: [https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-image/api](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-image/api)

Input and output images are downloaded and uploaded to fal before exposing to the Wan (fal) api and before returning the result to client. 

### Tests
The api was tested with Bruno and the files are included in the fal-bruno folder. You can find tests for all of the endpoints, but will have to change the url as the hash changes on different runs.

### AI Usage Disclaimer
All of the code was written by myself by reading the docs and github examples provided by fal, no AI was used for writing the app. 
AI was used inevitably in Google searches gemini responses for searching syntaxes.
After the code was finished and working, I asked Claude for feedback and improvement ideas, and replaced `print`s with `logger` and 
added `try-catches` in two functions that I had missed.

### Potential Improvements
In a real-world scenario like production, the jobs could be saved in the DB, and the queue can be on the cloud, e.g. AWS SQS, and 
a queue consumer can be used for execution.
Given more time, I would separate the classes in distinct files, I just wanted to avoid circular imports at this stage. 
I would also write automated tests for reliability and quality assurance.

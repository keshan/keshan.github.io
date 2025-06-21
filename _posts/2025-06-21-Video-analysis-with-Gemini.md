---
layout: post
title: "From Video to Structured Insight: A Practical Guide to Gemini-Powered Video Analysis"
date: 2025-06-21
description: "This guide demonstrates how to build a secure and reliable Python pipeline for video analysis using the Google Gemini API."
img: video_analysis.png
tags: [Gemini, Video Analysis, Computer Vision, LLM]
---

In today’s data-driven world, video content is everywhere—from security surveillance and industrial monitoring to social media and entertainment. Manually reviewing hours of footage is not only tedious but also prone to human error and inconsistency. This growing challenge has made automated video analysis an essential tool for organizations seeking timely, actionable insights from their video data.

Thanks to Gemini's multi-modal capabilities, it's now possible to harness large language models for comprehensive video analysis and insight extraction. This guide walks you through using the Google Gemini API to analyze video content and uncover valuable information. also we'll explore how to use pydantic and extract the final output in a structured format.


## Initial Setup

*   In order to setup, you must first create a Google Gemini API key. [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
*   Install the following Python packages: 
```bash 
pip install google-genai pydantic
```
*   Set the API key as an environment variable:
```bash
export GEMINI_API_KEY="your_api_key"
```
* You have to then import the necessary modules
```python
import os # This is necessary to get the environment variable

from google import genai # Google genai package to call Gemini API
from google.genai import types

from pydantic import BaseModel, Field # Pydantic will be used to structure the LLM output
from typing import Optional, List
```


In this initial setup, we have already created the API key, installed the necessary packages, and   imported the necessary modules.  

Now in this guide we will create a python script which will take a video file as input and return a summary and a list if anomalies in that video. for example if the video is of a factory, the anomalies could be if there's an unusual behaviour among the people who are working there or if there's an unusual activity in the factory. 

## Implementation

```python
class Anomaly(BaseModel):
    """Model for video anomalies detected during analysis"""
    description: str = Field(..., description="Description of the anomaly, be certain do not include any non-essential or unsure information")
    timestamp: str = Field(..., description="Timestamp when the anomaly occurs in the video")


class VideoAnalysisResult(BaseModel):
    """Model for structured video analysis results"""
    summary_of_the_video: str = Field(..., description="A detailed summary of the video content")
    anomalies: Optional[List[Anomaly]] = Field(None, description="List of anomalies detected in the video, if any")
```

this defines the structure of the output we expect from the LLM. Gemini will forced to give the final output in this structure. You must give an appropriate description for each field so that the LLM can understand what to generate.

Now, let's create a function to get a video path as input and call the Gemini API with the video and get the output in the structure we defined above.

```python
def analyze_video(video_path: str, prompt: str, model_name: str = "gemini-2.0-flash") -> VideoAnalysisResult:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    mime_type = "video/mp4"
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(
                        mime_type=mime_type,
                        data=video_bytes
                    )),
                    types.Part(text=prompt),
                ],
            ),
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": VideoAnalysisResult,
        }
    )
    return response.parsed
```
Above function takes a path to a video file, and a prompt which describes what we expect from the LLM given that video, and a Gemini model name which supports video format (at the moment of writing this article All Gemini 2.0 and 2.5 models can process video data). Apart from `mp4`, Gemini also supports other video formats like `mpeg, avi, webm, mov, mkv, x-flv, mpg, wmv, 3gpp`.

```python
prompt = """
You are a highly accurate video analysis model. Analyze the provided video and extract the following information **with high confidence only**:

- **1. Detailed Summary**  
  - Provide a clear, concise summary of the events in the video.  
  - Focus only on observable facts—avoid speculation.

- **2. Anomaly Detection**  
  - Identify any unusual, suspicious, or concerning events.  
  - For each anomaly, include:
    - A brief description  
    - Exact timestamp(s) where the anomaly occurs  
  - If **no anomalies** are confidently identified, return: `None`.

- **3. People Count**  
  - Count the total number of **unique individuals** visible at any point in the video.  
  - Only include people if clearly visible and identifiable as human.

**Strict Instructions:**  
- **Do not include** vague, speculative, or low-confidence observations.  
- **Only make assumptions** if they are **fully supported by visible evidence**.  
- If a field has **no certain data**, explicitly return `None` for that field.  
- Your analysis must prioritize **accuracy over completeness**.

"""
```

Now let's test the function with a video file.
```python
result = analyze_video("path_to_video.mp4", prompt)
print(result)
```
This will produce a structured Pydantic object which you have defined in the `VideoAnalysisResult` class. You can simply access the fields of the object to get the output of each field.

```python
print(result.summary_of_the_video)
print(result.anomalies)

for anomaly in result.anomalies:
    print(anomaly.description, anomaly.timestamp)

```
---

## Additional Notes

When you are using a video file which is larger than 20MB, you have to use the FilesAPI to first upload it and then send that file id to the Gemini API.

*Also, at the time of writing this article, Gemini 2.5 also works only when uploading video files with FilesAPI regardless of the size of the file.*

Let's adapt our analyze_video function to handle all these cases.

```python
def analyze_video(video_path: str, prompt: str, model_name: str = "gemini-2.0-flash") -> VideoAnalysisResult:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    mime_type = "video/mp4"

    file_size = os.path.getsize(video_path)

    # For larger videos (>20MB) or Gemini 2.5 models, use the Files API
    if file_size > 20 * 1024 * 1024 or "gemini-2.5" in model_name: 
        file = client.files.upload(
            file=video_path
        )
        file_id = file.name
        
        # After we upload the file We have to wait a bit 
        # until the uploaded file is processed.
        # Wait for file to become ACTIVE before using it

        max_attempts = 10  # Maximum number of attempts
        wait_time = 2  # Initial wait time in seconds
        attempt = 0
        
        while attempt < max_attempts:
            # Check file status
            for f in client.files.list():
                if f.name == file_id:
                    if f.state.name == "ACTIVE":
                        # Use the uploaded file object
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[f, prompt],
                            config={
                                "response_mime_type": "application/json",
                                "response_schema": VideoAnalysisResult,
                            }
                        )
                        break
            else:  
                attempt += 1
                import time
                time.sleep(wait_time)
                continue
            break  # Break out of the while loop if we broke out of the for loop
        
        if attempt >= max_attempts:
            raise TimeoutError(f"File {file_id} did not become ACTIVE after {max_attempts} attempts")
        
        
        # Cleanup: delete the uploaded file when done
        client.files.delete(name=file.name)
        
    else:
        # For smaller videos, or Gemini 2.0 models, 
        # we can send the video directly
        # Read video file into memory for inline data
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    parts=[
                        types.Part(inline_data=types.Blob(
                            mime_type=mime_type,
                            data=video_bytes
                        )),
                        types.Part(text=prompt),
                    ],
                ),
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": VideoAnalysisResult,
            }
        )    
    return response.parsed
```

Now this can handle any video file size supported by Gemini and any Gemini model that is capable of processing video data.

## Conclusion

This is a simple yet powerful example of how to use the Google Gemini API to analyze video content and extract valuable information. We have also explored how to use pydantic to structure the LLM output in a structured format.

A full implementation of this code can be found [here](https://github.com/keshan/gemini-video-analyzer)

## Next Steps

- Use this method to extract a dataset for a set of videos and then use that dataset to train/finetune a different model to predict the anomalies in the videos.
- Use Gemma3n like model and do the same on an edge device.
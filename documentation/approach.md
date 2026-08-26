# Initial Thoughts & Approach Breakdown

> [!NOTE]
> *These are my initial design thoughts and planning notes jotted down in Notion before starting implementation.*

---

## 1. Video & Audio Observations

The video begins with credits — on-screen text acknowledging the author, a couple of actors, and the title of the video — with no audio. This may or may not be present in other videos that may be used to test the generalisation of the model, but it gives me a case to work on. 

The second case is the video itself contains dialogues spoken throughout the duration with some silences in between. There are no transcripts provided and no on-screen captions either. So I am inclined to use an ASR model (Automatic Speech Recognition). 

Another point I noticed while watching the video to observe for cases to model for is that since the video is from an episode of *The Adventures of Sherlock Holmes*, there is a narrator (Watson) apart from other dialogues that happen between multiple people, Watson included.

---

## 2. Core Task & Deliverables

The task is to identify the exact frame in the video that a dialogue appears, given an input text `dialogue`. 

The model must return:
1. **Timestamp**: The timestamp (till milliseconds).
2. **Frame Number**: The frame number in which the dialogue first appears (since dialogue could be more than 2 frames long).
3. **Extracted Dialogue**: The dialogue extracted by the model.
4. **Video Frame Image**: The identified video frame as an image.

> [!TIP]
> *UI Idea:* I think I can also display the text that occurs in just the 1 frame where the model identifies the dialogue to first appear.

### Ground Truth Observation
While watching the video to observe for the cases to model for, I noticed that the dialogue to be found lies at the **~5.25 - 5.28** timestamp. This could be used for checking if the model is obtaining the correct output.

---

## 3. Key Design Considerations

- **Robustness**: Robust to variations in video quality, resolution, and frame rate.
- **Appearance**: Robust to "appearance of the dialogue" (pauses, audio cadence).
- **Resolution Independence**: The model should work for any kind of video irrespective of the upload format, and performance should not deteriorate for lower video resolutions.

---

## 4. Interface & User Experience

Once the model is created and tested out, I can work more on the interface. 

- For the interface itself, I’m thinking of having a **Streamlit** or **Gradio** site which uses the deployed model to return the required outputs.
- I’d like to have a text input `URL`, which can take the URL of the video and use that as the input for the model.
- Also have a file upload option to load videos from the local system.

---

## 5. Model Approach & Experimentation Plan

The approach for the model I’m thinking is that I’ll use an ASR model to obtain the full text with timestamps. I can then iterate through this struct to find the input dialogue, and display the other relevant information. With the timestamp now in hand, I can extract the specific frame from the video using the timestamp.

### Thoughts on Extensibility & Edge Cases

- **Approach 1 (Exact Search)**: Works only if the input `dialogue` is present exactly in the video, else it returns NULL output. But the test input dialogue could be something semantically similar to the dialogue but not exactly, so ensuring the model isn’t highly sensitive to the exact text could also be useful. 
- **Approach 2 (Embeddings + Minimum Edit Distance)**: I could think about converting the transcript generated from ASR to embeddings, and also the user’s query to embeddings, and then performing a similarity search on the embeddings to thereby identify the corresponding frame. Another thing I can factor in using concepts learned in NLP class is to use Minimum edit distance algorithms to identify typos in the input `dialogue`.
- **Approach 3 (Multimodal Vision-Language Model)**: Use an existing multimodal model like Qwen VL for the entire pipeline mentioned in approach 2. 

> [!IMPORTANT]
> *Experiment Workflow:* I can compare results obtained from all 3 and then choose the best out of them all as the final model, keeping the 3 experiments in a `notebooks/` folder in the github repo.

### Accent Handling
Also have to think about how any approach will handle ASR if the English is in different accents (e.g., Indian, Chinese, Singaporean, etc.) — think if there’s something that can auto detect accent and accordingly choose the ASR model that performs best for that accent.

---

> [!NOTE]
> 💡 **Next Steps**:
> 1. Need to write unit tests.
> 2. Start off with a script to load the video from the URL.

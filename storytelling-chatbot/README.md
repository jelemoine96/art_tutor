[![Try](https://img.shields.io/badge/try_it-here-blue)](https://storytelling-chatbot.fly.dev)

# Storytelling Chat Bot

<img src="frontend/app/opengraph-image.png" width="420px">

This example shows how to build a voice-driven interactive storytelling experience.
It periodically prompts the user for input for a 'choose your own adventure' style experience.

We add visual elements to the story by generating images at lightning speed using Fal.

<img src="image.png" width="420px">

---

### It uses the following AI services:

**Deepgram - Speech-to-Text**

Transcribes inbound participant voice media to text.

**OpenAI (GP4 3) - LLM**

Our creative writer LLM. You can see the context used to prompt it [here](src/prompts.py)

**ElevenLabs - Text-to-Speech**

Converts and streams the LLM response from text to audio

**Fal.ai - Image Generation**

Adds pictures to our story (really fast!) Prompting is quite key for style consistency, so we task the LLM to turn each story page into a short image prompt.

---

## Setup

**Install requirements**

```shell
pip install -r requirements.txt
```

**Create environment file and set variables:**

```shell
mv env.example .env
```

**Build the frontend:**

This project uses a custom frontend, which needs to built. Note: this is done automatically as part of the Docker deployment.

```shell
cd frontend/
npm install / yarn
npm run build
```

The build UI files can be found in `frontend/out`

## Running it locally

Start the API / bot manager:

`python src/server.py`

If you'd like to run a custom domain or port:

`python src/server.py --host somehost --p 7777`

➡️ Open the host URL in your browser

> [!IMPORTANT]
> Whilst working on the frontend code, please `yarn run dev`
> and open the NextJS hosted service vs. the Python server.
> (Usually localhost:3000.)

---

## How does it work?

todo

---

## Deploying

WIP

## Improvements to make

- Wait for track_started event to avoid rushed intro
- Show 5 minute timer on the UI

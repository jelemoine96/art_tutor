import asyncio

import aiohttp
import logging
import os
from PIL import Image
from typing import AsyncGenerator

from openai import AsyncOpenAI, AsyncStream
from PIL import Image
import io
import time
import base64
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from dailyai.pipeline.aggregators import (
    LLMUserResponseAggregator,
    ParallelPipeline,
    VisionImageFrameAggregator,
    SentenceAggregator
)
from dailyai.pipeline.frames import (
    ImageFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    VisionImageFrame,
    AudioFrame,
    PipelineStartedFrame,
    TTSEndFrame,
    TextFrame,
    UserImageFrame,
    UserImageRequestFrame,
)
from dailyai.services.moondream_ai_service import MoondreamService
from dailyai.pipeline.pipeline import FrameProcessor, Pipeline
from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.ai_services import VisionService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

user_request_answer = "Let me take a look."

sprites = []

script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(img.tobytes())

flipped = sprites[::-1]
sprites.extend(flipped)

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = ImageFrame(sprites[0], (1024, 576))
talking_frame = SpriteFrame(images=sprites)


class TalkingAnimation(FrameProcessor):
    """
    This class starts a talking animation when it receives an first AudioFrame,
    and then returns to a "quiet" sprite when it sees a LLMResponseEndFrame.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, AudioFrame):
            if not self._is_talking:
                yield talking_frame
                yield frame
                self._is_talking = True
            else:
                yield frame
        elif isinstance(frame, TTSEndFrame):
            yield quiet_frame
            yield frame
            self._is_talking = False
        else:
            yield frame


class AnimationInitializer(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PipelineStartedFrame):
            yield quiet_frame
            yield frame
        else:
            yield frame

artist="""    ###Artist rules###
    You act as the artist who created the picture
    You start by introducing who you are using the specific name of the artist and giving some brief explanation about the artwork.
    You only expand if asked
    Keep responses short
    If we say 'ENOUGH', you go back to being an Art Curator and you say 'Its you curator here again'
    Since your presentation will be delivered via an audio guide, avoid using any special characters in your responses. 
    Your aim is to engage the listener by providing creative yet succinct explanations, drawing attention to the nuances of each artwork. 
    
    """

class UserImageRequester(FrameProcessor):
    participant_id: str | None

    def __init__(self):
        super().__init__()
        self.participant_id = None

    def set_participant_id(self, participant_id: str):
        self.participant_id = participant_id

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if self.participant_id and isinstance(frame, TextFrame):
            if frame.text == user_request_answer:
                yield UserImageRequestFrame(self.participant_id)
                yield TextFrame(artist)
        elif isinstance(frame, UserImageFrame):
            yield frame

class OpenAIVisionService(VisionService):
    def __init__(
        self,
        *,
        model="gpt-4-vision-preview",
        api_key,
    ):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def run_vision(self, frame: VisionImageFrame):
        IMAGE_WIDTH = frame.size[0]
        IMAGE_HEIGHT = frame.size[1]
        # COLOR_FORMAT = "RGBA"
        print("IMAGE dimensions", IMAGE_HEIGHT, IMAGE_WIDTH)
        a_image = Image.frombytes(
            'RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), frame.image)
        # new_image = a_image.convert('RGB')

        # Uncomment these lines to write the frame to a jpg in the same directory.
        # current_path = os.getcwd()
        # image_path = os.path.join(current_path, "image.jpg")
        # image.save(image_path, format="JPEG")
        print("Inside run_vision")
        jpeg_buffer = io.BytesIO()

        a_image.save(jpeg_buffer, format='JPEG')
        print("run vision")
        jpeg_bytes = jpeg_buffer.getvalue()
        print("JPEG bytes", jpeg_bytes)
        base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
        print("Base64")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": frame.text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]
        print(messages)
        chunks: AsyncStream[ChatCompletionChunk] = (
            await self._client.chat.completions.create(
                model=self._model,
                stream=True,
                messages=messages,
            )
        )
        print("Chunks returned", chunks)
        result = ""
        async for chunk in chunks:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content:
                result += " " + chunk.choices[0].delta.content

        return result


class TextFilterProcessor(FrameProcessor):
    text: str

    def __init__(self, text: str):
        self.text = text

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TextFrame):
            if frame.text != self.text:
                yield frame
        else:
            yield frame


class ImageFilterProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if not isinstance(frame, ImageFrame):
            yield frame


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=True,
            camera_width=1024,
            camera_height=576,
            vad_enabled=True,
            video_rendering_enabled=True
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="8xmZutR0C1ug1qQSU5de",
        )


        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview")

        ta = TalkingAnimation()
        ai = AnimationInitializer()

        sa = SentenceAggregator()
        ir = UserImageRequester()
        va = VisionImageFrameAggregator()
        # If you run into weird description, try with use_cpu=True
        # moondream = MoondreamService()
        vision = OpenAIVisionService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model = "gpt-4-vision-preview"
        )

        tf = TextFilterProcessor(user_request_answer)
        imgf = ImageFilterProcessor()

        prompt = f"""
    You have to role you are either an art curator or a famous artist. Always act in accordance in being in one of those roles. 
    ###RULES###
    1. You start by being art curator
    2. If you recognise a picture, you become a famous artist and start speaking as you are the artist
    ###Art Curator rules###
    If the user asks you to describe a specific piece of art, respond concisely with '{user_request_answer}' and change to artist role. 
    Since your presentation will be delivered via an audio guide, avoid using any special characters in your responses. 
    Your aim is to engage the listener by providing creative yet succinct explanations, drawing attention to the nuances of each artwork. 
    Always keep your responses as brief as possible. 1 to 2 sentences maximum
    ###Artist rules###
    You act as the artist who created the picture
    You start by giving some brief explanation about when you drew it.
    You only expand if asked
    Keep responses short
    If we say 'ENOUGH', you go back to being an Art Curator and you say 'Its you curator here again'
    Since your presentation will be delivered via an audio guide, avoid using any special characters in your responses. 
    Your aim is to engage the listener by providing creative yet succinct explanations, drawing attention to the nuances of each artwork. 
    Always keep your responses as brief as possible. 1 to 2 sentences maximum
    """

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
        ]

        ura = LLMUserResponseAggregator(messages)

        pipeline = Pipeline([
            ai, ura, llm, ParallelPipeline(
                [[sa, ir, va, vision], [tf, imgf]]
            ),
            tts, ta
        ])

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport, participant):
            transport.render_participant_video(participant["id"], framerate=0)
            ir.set_participant_id(participant["id"])
            await pipeline.queue_frames([LLMMessagesFrame(messages)])

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True

        await asyncio.gather(transport.run(pipeline))


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

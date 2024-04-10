import React, { useState } from "react";
import {
  useDaily,
  useParticipantIds,
  useAppMessage,
  DailyAudio,
} from "@daily-co/daily-react";
import VideoTile from "@/components/VideoTile";
import { Button } from "@/components/ui/button";
import WaveText from "@/components//WaveText";
import UserInputIndicator from "@/components/UserInputIndicator";
import { IconLogout } from "@tabler/icons-react";

interface StoryProps {
  handleLeave: () => void;
}

const Story: React.FC<StoryProps> = ({ handleLeave }) => {
  const daily = useDaily();
  const participantIds = useParticipantIds({ filter: "remote" });
  const [storyState, setStoryState] = useState<"user" | "assistant">(
    "assistant"
  );

  useAppMessage({
    onAppMessage: (e) => {
      if (!daily) return;

      /*if (e.fromId === "transcription") {
        console.log(e.data?.text);
      }*/

      if (!e.data?.cue) return;

      if (e.data?.cue === "user_turn") {
        daily.setLocalAudio(true);
        setStoryState("user");
      } else {
        daily.setLocalAudio(false);
        setStoryState("assistant");
      }
    },
  });

  return (
    <div className="w-full flex flex-col flex-1 self-stretch">
      <div className="absolute top-20 w-full z-20 text-center">
        <WaveText active={storyState === "user"} />
      </div>

      <header className="flex absolute top-0 w-full z-50 p-6 justify-end">
        <Button variant="secondary" onClick={() => handleLeave()}>
          <IconLogout size={21} className="mr-2" />
          Exit
        </Button>
      </header>
      <div className="absolute inset-0 bg-gray-900 bg-opacity-80 z-10 fade-in"></div>

      <div className="relative z-20 flex-1">
        {participantIds.length ? (
          <VideoTile sessionId={participantIds[0]} />
        ) : (
          <div>Loading</div>
        )}

        <DailyAudio />
      </div>
      <UserInputIndicator active={storyState === "user"} />
    </div>
  );
};

export default Story;

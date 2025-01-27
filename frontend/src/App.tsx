import { useState } from "react";
import Recommend from "./Recommend";
import { Anime } from './types/Anime';
import { SelectedAnime } from "./types/SelectedAnime";
import { Button } from "./components/ui/button";

const App = () => {
  const [recommends, updateRecommends] = useState<Anime[]>([]);
  const [selectedAnime, setSelectedAnime] = useState<SelectedAnime[]>([]);
  const recommendFunction = async () => {
    const ratings: [string, number][] = selectedAnime.map((anime) => [anime.anime_id, anime.rating]);

    console.log("Sent", ratings);

    try {
      const response = await fetch("http://127.0.0.1:5000/recommendations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ratings }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Recommendations from Flask:", data);
        updateRecommends(data);
      } else {
        console.error("Error while fetching recommendations");
      }
    } catch (error) {
      console.error("Error while making the request:", error);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col justify-center items-center bg-gray-900">
      {recommends.length === 0 ? (
        <Recommend
          recommendFunctionClick={recommendFunction}
          selectedAnime={selectedAnime}
          setSelectedAnime={setSelectedAnime}
        />
      ) : (
        <div className="flex flex-col items-center text-white">
          <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
          <ul className="space-y-2">
            {recommends.map((recommend, index) => (
              <li key={index} className="text-lg">{recommend.name}</li>
            ))}
          </ul>
          <Button className="mt-20" onClick={(() => updateRecommends([]))}>Clear</Button>
        </div>
      )}
    </div>
  );
};

export default App;

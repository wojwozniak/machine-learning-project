import { useState } from "react";
import Recommend from "./Recommend";
import { SelectedAnime } from "./types/SelectedAnime";
import { Button } from "./components/ui/button";

const App = () => {
  const [recommends, updateRecommends] = useState<any[]>([]);
  const [selectedAnime, setSelectedAnime] = useState<SelectedAnime[]>([]);
  const recommendFunction = async () => {
    const ratings: [number, number][] = selectedAnime.map((anime) => [Number(anime.anime_id), anime.rating]);

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
          <table className="min-w-full table-auto border-collapse">
            <thead>
              <tr>
                <th className="px-4 py-2 border">Name</th>
                <th className="px-4 py-2 border">Rating</th>
                <th className="px-4 py-2 border">Model Score</th>
              </tr>
            </thead>
            <tbody>
              {recommends.map((recommend, index) => (
                <tr key={index}>
                  <td className="px-4 py-2 border">{recommend.name}</td>
                  <td className="px-4 py-2 border">{recommend.rating}</td>
                  <td className="px-4 py-2 border">{recommend.score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Button className="mt-20" onClick={(() => updateRecommends([]))}>Clear</Button>
        </div>
      )}
    </div>
  );
};

export default App;

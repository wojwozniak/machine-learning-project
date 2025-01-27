import { useState } from "react";
import Recommend from "./Recommend";
import { SelectedAnime } from "./types/SelectedAnime";

const App = () => {
  const [recommends, updateRecommends] = useState<string[]>([]);
  const [selectedAnime, setSelectedAnime] = useState<SelectedAnime[]>([]);
  const recommendFunction = async () => {
    const animeTuples: [string, number][] = selectedAnime.map((anime) => [anime.anime_id, anime.rating]);

    console.log('Sent', animeTuples);

    try {
      const response = await fetch('http://127.0.0.1:5000/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ animeTuples })
      });

      console.log(response);

      if (response.ok) {
        const data = await response.json();
        console.log('Recommendations from Flask:', data);
      } else {
        console.error('Error while fetching recommendations');
      }
    } catch (error) {
      console.error('Error while making the request:', error);
    }
  }

  return (
    <div>
      {recommends.length == 0 ? (
        <Recommend recommendFunctionClick={recommendFunction}
          selectedAnime={selectedAnime}
          setSelectedAnime={setSelectedAnime} />
      ) : (
        <div>test</div>
      )}
    </div>
  );
};

export default App;

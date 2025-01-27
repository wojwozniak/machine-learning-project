import { useState, useEffect } from 'react';
import { Search, Trash2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import jsonFile from './../../data/anime.json';
import { Anime } from './types/Anime';
import { SelectedAnime } from './types/SelectedAnime';

interface RecommendProps {
    recommendFunctionClick: () => void;
    selectedAnime: SelectedAnime[];
    setSelectedAnime: React.Dispatch<React.SetStateAction<SelectedAnime[]>>;
}

const Recommend: React.FC<RecommendProps> = ({ recommendFunctionClick, selectedAnime, setSelectedAnime }) => {
    const [animeData, setAnimeData] = useState<Anime[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [isSearchFocused, setIsSearchFocused] = useState(false);

    useEffect(() => {
        setAnimeData(jsonFile);
    }, []);

    const filteredAnime = animeData.filter(anime =>
        anime.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleSelect = (anime: Anime) => {
        if (!selectedAnime.some(selected => selected.anime_id === anime.anime_id)) {
            setSelectedAnime([...selectedAnime, { ...anime, rating: 0 }]);
        }
        setSearchQuery('');
        setIsSearchFocused(false);
    };

    const handleDelete = (animeId: string) => {
        setSelectedAnime(selectedAnime.filter(anime => anime.anime_id !== animeId));
    };

    const handleRating = (animeId: string, rating: number) => {
        setSelectedAnime(selectedAnime.map(anime =>
            anime.anime_id === animeId ? { ...anime, rating } : anime
        ));
    };

    const handleClick = () => {
        recommendFunctionClick();
    };

    return (
        <div className="h-screen w-screen bg-gray-900 text-white">
            <div className="max-w-2xl mx-auto p-6">
                <div className="relative mb-8">
                    <div className="relative">
                        <Input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onFocus={() => setIsSearchFocused(true)}
                            placeholder="Search anime..."
                            className="pl-10 bg-gray-800 border-gray-700 text-white placeholder-gray-400 focus:ring-blue-500 focus:border-blue-500"
                        />
                        <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                    </div>

                    {isSearchFocused && searchQuery && (
                        <Card className="absolute w-full mt-1 z-10 bg-gray-800 border-gray-700">
                            <CardContent className="p-2">
                                {filteredAnime.length > 0 ? (
                                    filteredAnime.map(anime => (
                                        <button
                                            key={anime.anime_id}
                                            onClick={() => handleSelect(anime)}
                                            className="w-full text-left p-2 text-white hover:bg-gray-700 rounded"
                                        >
                                            {anime.name}
                                        </button>
                                    ))
                                ) : (
                                    <div className="p-2 text-gray-400">No results found</div>
                                )}
                            </CardContent>
                        </Card>
                    )}
                </div>

                <div className="space-y-4">
                    {selectedAnime.map(anime => (
                        <Card key={anime.anime_id} className="bg-gray-800 border-gray-700">
                            <CardContent className="flex items-center justify-between p-4">
                                <div className="flex-1">
                                    <h3 className="font-medium text-white">{anime.name}</h3>
                                    <div className="flex items-center mt-2 flex-wrap gap-1">
                                        {[...Array(10)].map((_, index) => (
                                            <Button
                                                key={index}
                                                variant={anime.rating === index + 1 ? "default" : "outline"}
                                                size="sm"
                                                className={`p-1 min-w-8 h-8 ${anime.rating === index + 1
                                                    ? 'bg-blue-600 hover:bg-blue-700'
                                                    : 'border-gray-600 text-gray-300 hover:bg-gray-700'
                                                    }`}
                                                onClick={() => handleRating(anime.anime_id, index + 1)}
                                            >
                                                {index + 1}
                                            </Button>
                                        ))}
                                    </div>
                                </div>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="ml-4 text-gray-400 hover:text-white hover:bg-gray-700"
                                    onClick={() => handleDelete(anime.anime_id)}
                                >
                                    <Trash2 className="h-5 w-5" />
                                </Button>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>
            <div className="max-w-2xl mx-auto p-6">
                <div className="relative mb-8">
                    <div className="relative">
                        {selectedAnime.length > 0 && <Button onClick={handleClick}>Recommend</Button>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Recommend;
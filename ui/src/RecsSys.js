import { useState } from "react";
import { useCallback } from "react";
import { About } from "./components/About";
import { Banner } from "./components/Banner";

import { ParallaxBackground } from "./components/ParallaxBackground";

import { SearchResults } from "./components/SearchResults";
import { SearchRecsClient } from "./utils/SearchRecsClient";

const showAbout = true;

export function RecSys() {
  const { movieIds, moviesLoading, onSearchRecommendations, onResetSearch } =
    useHooks();
  return (
    <>
      {showAbout && <About />}
      <Banner
        onSearchRecommendations={onSearchRecommendations}
        onResetSearch={onResetSearch}
      />
      <SearchResults movieIds={movieIds} loading={moviesLoading} />
      <div style={{ marginBottom: "4rem" }}>
        <ParallaxBackground />
      </div>
    </>
  );
}

function useHooks() {
  const [movieIds, setMovieIds] = useState(undefined);
  const [moviesLoading, setMoviesLoading] = useState(false);

  const onSearchRecommendations = useCallback((req) => {
    const client = new SearchRecsClient();
    setMovieIds(undefined);
    setMoviesLoading(true);
    client.fetchRecommendationsForRequest(req).then((res) => {
      if (res["movieIds"]) {
        setMovieIds(res["movieIds"]);
        setMoviesLoading(false);
      }
    });
  }, []);

  const onResetSearch = useCallback(() => {
    setMovieIds(undefined);
    if (moviesLoading) setMoviesLoading(false);
  }, [moviesLoading]);

  return {
    movieIds,
    moviesLoading,
    onSearchRecommendations,
    onResetSearch,
  };
}

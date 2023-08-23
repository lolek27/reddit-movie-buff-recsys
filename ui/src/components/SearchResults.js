//create a component called SearchResults which will contain the results of the search
import React, { useEffect } from "react";
import MovieCard from "./MovieCard";
import { MoviesLoading } from "./MoviesLoading";

export const SearchResults = ({ movieIds, loading }) => {
  const setResultsOffset = () => {
    const scrollDiv = document.querySelector(".search-results");
    const bannerDiv = document.querySelector(".banner");
    console.log(
      `Height: ${bannerDiv.offsetHeight} and offset: ${bannerDiv.offsetTop}`
    );
    if (bannerDiv.offsetHeight && !isNaN(bannerDiv.offsetTop)) {
      scrollDiv.style.height = `calc(100% - 3rem - ${
        bannerDiv.offsetHeight + bannerDiv.offsetTop
      }px`;
    } else {
      scrollDiv.classList.add("high");
    }
  };

  useEffect(() => {
    if (loading) {
      let scrolled = false;

      window.addEventListener("scroll", () => {
        if (!scrolled) {
          scrolled = true;
          return;
        }
        setResultsOffset();
      });
      setResultsOffset();
    }
    //    if (window.scrollY > window.innerHeight / 2) {
    //         console.log(window.scrollY, window.innerHeight)
    //        scrollDiv.classList.add('sticky');
    //    } else {
    //        scrollDiv.classList.remove('sticky');
    //    }
  }, [loading]);
  return (
    <div className="search-results pt-3">
      {!!movieIds &&
        movieIds.map((mid) => (
          <MovieCard key={mid} id={mid} className="mb-4" />
        ))}
      {!movieIds && loading && <MoviesLoading />}
    </div>
  );
};

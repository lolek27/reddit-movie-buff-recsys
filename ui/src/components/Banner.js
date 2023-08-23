import React, { useCallback, useEffect, useState } from "react";

import reddit from "../assets/reddit-logo512.png";
import { RecSysInput } from "./RecSysInput";

export function Banner({ onSearchRecommendations, onResetSearch }) {
  const [value, setValue] = useState("");

  const scrollToSearchInput = () => {
    window.scrollTo(0, 0);
  };

  const onChange = (val) => {
    console.log("onChange", val);
    if (!val) {
      onResetSearch();
      scrollToSearchInput();
    }
    setValue(val);
  };

  const onSearchResults = useCallback(
    (recommendationRequest) => {
      if (recommendationRequest) {
        console.log("onSearchResults", recommendationRequest);
        onSearchRecommendations(recommendationRequest);
      }
    },
    [onSearchRecommendations]
  );

  useEffect(() => {
    const scrollDiv = document.querySelector(".banner");
    let scrolled = false;

    window.addEventListener("scroll", () => {
      if (!scrolled) {
        scrolled = true;
        return;
      }
      scrollDiv.classList.add("sticky");
    });
  });

  return (
    <div className="banner">
      <div className="search-banner">
        <div className="recsys-title">
          <span>Reddit Movie Buff</span>
          <span className="icon">
            <a href="https://www.reddit.com/" target="_blank">
              <img src={reddit} width={60} height={60} alt="reddit logo" />
            </a>
          </span>
        </div>
        <RecSysInput
          className="mt-2"
          value={value}
          onChange={onChange}
          onSearchResults={onSearchResults}
        />
      </div>
    </div>
  );
}

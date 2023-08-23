import { useCallback } from "react";
import { useEffect, useState } from "react";
import Button from "react-bootstrap/Button";
import Card from "react-bootstrap/Card";
import { TmdbClientWrapper } from "../utils/TmdbClient";
import { LoadingSkeleton } from "./LoadingSkeleton";
import tmdb from "../assets/tmdb.png";
import imdb from "../assets/imdb.png";

const MovieButton = ({
  id,
  type = "tmdb",
  title = "TMDB",
  icon,
  className,
}) => {
  let url = `https://www.themoviedb.org/movie/${id}`;
  if (type === "imdb") {
    url = `https://www.imdb.com/title/${id}`;
  }
  const goTo = useCallback((url) => {
    window.open(url, "_blank");
  }, []);

  return (
    <Button
      className="db-button"
      onClick={() => goTo(url)}
      variant="outline-secondary">
      <img
        src={icon}
        width={type === "tmdb" ? 30 : 34}
        height={type === "tmdb" ? 30 : 28}
        alt={type}
      />
      <span style={{ paddingLeft: "5px" }}>
        Find in <span className="db-title">{title}</span>
      </span>
    </Button>
  );
};

function MovieCard({ id, className }) {
  const [movieDetails, setMovieDetails] = useState(null);

  const processMovieInfo = useCallback(
    (movieInfo) => {
      const div = document.getElementById(id);
      if (div) {
        div.style.backgroundImage = `url('${movieInfo.backdropUrl}')`;
        div.style.backgroundSize = "cover";
        div.style.backgroundRepeat = "no-repeat";
        div.style.backgroundPosition = "left top";
      }
    },
    [id]
  );

  useEffect(() => {
    const client = new TmdbClientWrapper();
    client.fetchMovieInfoById(id).then((movieInfo) => {
      // console.log(movieInfo);
      setMovieDetails(movieInfo);
      processMovieInfo(movieInfo);
    });
  }, [id, processMovieInfo]);

  if (!movieDetails) return <LoadingSkeleton className="mb-4" />;

  return (
    <div id={id} className={"movie-card " + className}>
      {movieDetails && (
        <Card className="backdrop">
          <div className="body">
            <img
              className="movie-img"
              src={movieDetails.posterUrl}
              alt={movieDetails.title}
            />
            <div className="content">
              <h3 className="title">
                {movieDetails.title}{" "}
                <span className="title-year">
                  ({movieDetails.releaseDate.year})
                </span>
              </h3>
              <div className="subtitle">
                <span className="release-date">
                  {movieDetails.releaseDate.releaseDate}
                </span>
                <span className="country">{" " + movieDetails.country}</span>
                <span className="genres">{movieDetails.genres}</span>
              </div>
              {movieDetails.tagline && (
                <div className="tagline">{movieDetails.tagline}</div>
              )}

              <div className="overview">
                <div>Overview:</div>
                <p>{movieDetails.overview}</p>
              </div>
              <div className="db-buttons">
                <MovieButton id={movieDetails.id} icon={tmdb} />
                <MovieButton
                  id={movieDetails.imdb_id}
                  title="IMDb"
                  type="imdb"
                  icon={imdb}
                />
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

export default MovieCard;

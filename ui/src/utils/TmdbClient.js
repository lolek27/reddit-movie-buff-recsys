const TMDB_API_KEY = process.env.REACT_APP_TMDB_API_KEY;

export class TmdbClientWrapper {
  constructor() {
    this.apiKey = TMDB_API_KEY;
  }

  // Define a function to fetch movie information and poster by movie name
  async fetchMovieInfoByName(movieName) {
    // Fetch the movie ID
    const searchUrl = `https://api.themoviedb.org/3/search/movie?api_key=${this.apiKey}&query=${movieName}`;
    const searchResponse = await fetch(searchUrl);
    const searchResult = await searchResponse.json();
    const movieId = searchResult.results[0].id;

    // Fetch the movie details by ID
    return this.fetchMovieInfoById(movieId);
  }

  // Define a function to fetch movie information and poster by movie ID
  async fetchMovieInfoById(movieId) {
    // Fetch the movie details
    const detailsUrl = `https://api.themoviedb.org/3/movie/${movieId}?api_key=${this.apiKey}&language=en-US`;
    const detailsResponse = await fetch(detailsUrl);
    const detailsResult = await detailsResponse.json();

    // Fetch the movie poster
    const posterUrl = `https://image.tmdb.org/t/p/w500/${detailsResult.poster_path}`;
    let posterResponse = await fetch(posterUrl);
    const posterBlob = await posterResponse.blob();
    const posterUrlObject = URL.createObjectURL(posterBlob);

    // Fetch the movie backdrop
    const backdropUrl = `https://image.tmdb.org/t/p/w1920_and_h800_multi_faces/${detailsResult.backdrop_path}`;
    posterResponse = await fetch(backdropUrl);
    const backBlob = await posterResponse.blob();
    const backdropUrlObject = URL.createObjectURL(backBlob);

    // Return the movie information and poster URL
    return {
      id: detailsResult.id,
      imdb_id: detailsResult.imdb_id,
      title: detailsResult.title,
      overview: detailsResult.overview,
      releaseDate: this.formatDate(detailsResult.release_date),
      posterUrl: posterUrlObject,
      backdropUrl: backdropUrlObject,
      genres: this.getGenres(detailsResult.genres),
      country: this.getCountry(detailsResult.production_countries),
      tagline: detailsResult.tagline,
    };
  }

  formatDate(dateString) {
    if (dateString) {
      const [year, month, day] = dateString.split("-");
      return { releaseDate: `${month}/${day}/${year}`, year };
    }
    return {};
  }
  getGenres(genres) {
    return genres.map((g) => g.name).join(", ");
  }
  getCountry(countries) {
    if (countries.length > 0) {
      return `(${countries[0]["iso_3166_1"]})`;
    }
    return "";
  }
}

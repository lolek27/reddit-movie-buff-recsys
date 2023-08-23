const SEARCH_URL = "http://100.26.188.128/search_recs/";
// "http://127.0.0.1:8000/search_recs/";

export class SearchRecsClient {
  constructor() {
    this.baseUrl = SEARCH_URL;
  }

  async fetchRecommendationsForRequest(req) {
    const data = { req };
    const response = await fetch(this.baseUrl, {
      method: "POST",
      mode: "cors",
      cache: "no-cache",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
      },
      redirect: "follow",
      referrerPolicy: "no-referrer",
      body: JSON.stringify(data),
    });
    return response.json(); // parses JSON response
  }
}

// wrapper class around localStorage with get and set functions for adding and retrieving data
export class LocalStorageWrapper {
  constructor() {
    this.storage = window.localStorage;
  }
  set(key, value) {
    this.storage.setItem(key, value);
  }
  get(key) {
    return this.storage.getItem(key);
  }
}

//class called Autocomplete that will take a string and check if it is in the local storage under key 'searches'
//if it is, it will return all values from local storage that start with the string
//if it is not, it will return an empty array and add the string to local storage under key 'searches'
export class LocalStorage {
  constructor() {
    this.storage = new LocalStorageWrapper();
    this.searchKey = "searches";
  }
  getResults(phrase) {
    let results = [];
    if (this.storage.get(this.searchKey)) {
      results = JSON.parse(this.storage.get(this.searchKey)).filter((item) =>
        item.includes(phrase)
      );
    }
    return results;
  }
  getAllResults() {
    let results = [];
    if (this.storage.get(this.searchKey)) {
      results = JSON.parse(this.storage.get(this.searchKey));
    }
    return results.filter((r) => typeof r === "string");
  }

  addResult(phrase) {
    let results = [];
    if (this.storage.get(this.searchKey)) {
      results = JSON.parse(this.storage.get(this.searchKey));
    }

    if (!results.includes(phrase)) {
      // if there are no results containing the phrase, add it
      results = this.getAllResults();
      results.push(phrase);
      this.storage.set(this.searchKey, JSON.stringify(results));
    }
  }
}

//create comoonent called browserInput which will contain autocomplete input
import React, { useCallback, useState } from "react";
import { Form, InputGroup, Button } from "react-bootstrap";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
// import { Typeahead } from "react-bootstrap-typeahead";
import reddit from "../assets/reddit-logo128.png";
import { LocalStorage } from "../utils/LocalStorage";
// import { SearchResults } from './SearchResults';
// import "/node_modules/react-bootstrap-typeahead/css/Typeahead.css";

const defaultOptions = [
  `Looking for Mysteries that take place in one room/building/setting. I'm looking for movies where it's a couple people (or more) 
in a room or small place where you can't really tell what's going on or what is going to happen until the end`,
  `Vampire/Monsters movies. Would like some reccomendations of a good creature movie. Preference to vampires.`,
  `Action Movies without guns. I saw John Wick 4 in the theater. I really enjoyed it, but holy shit there was more bullets shot than there were seconds in the almost 2 5/6 hour movie!
Got me wondering - what good action movie (preferably more modern day) that doesn't rely on 50,000 bullets?`,
  `I want to watch romantic movies that feature a depressed girl and a sunshine boy instead of the other way around
I know the title sounds really stupid but pop culture has always had the depressed boy and sunshine girl trope and I'm tired of the same stuff over and over again. 
As a depressed girl, I'd like to watch a few movies that have the opposite trope just to comfort myself.`,
];

function createDefaultOptions() {
  const ac = new LocalStorage();
  const allOptions = [...ac.getAllResults(), ...defaultOptions];
  if (allOptions.length) {
    return Array.from(new Set(allOptions)); //.map((o) => ({ id: o, label: o }));
  }
  return [];
}

const RecAutocomplete = ({ value, onChange, onSearchResults, className }) => {
  const [options, setOptions] = useState(createDefaultOptions());

  const handleSelect = useCallback(
    (ev) => {
      const txt = ev.target.outerText;
      onChange(txt);
    },
    [onChange]
  );

  const handleInputChange = useCallback(
    (ev) => {
      const txt = ev.target.value;
      if (txt !== 0) onChange(txt);
    },
    [onChange]
  );

  const onClickButton = useCallback(() => {
    console.log(`Searching for ${value}`);
    new LocalStorage().addResult(value);
    onSearchResults(value);
    setOptions(createDefaultOptions());
  }, [value, onSearchResults]);

  return (
    <Form.Group className={"browser-input " + className}>
      <InputGroup>
        <Autocomplete
          freeSolo
          options={options}
          onChange={handleSelect}
          onInputChange={handleInputChange}
          renderInput={(params) => (
            <TextField
              multiline
              {...params}
              value={value}
              label=""
              InputProps={{
                ...params.InputProps,
                placeholder: "🍿 What kind of movie are you looking for?",
                type: "search",
              }}
            />
          )}
        />

        <Button
          id="go-search-button"
          className="search-button"
          onClick={onClickButton}
          variant="outline-secondary">
          <img src={reddit} width={32} height={32} />
          <span style={{ paddingLeft: "5px" }}>Search</span>
        </Button>
      </InputGroup>
    </Form.Group>
  );
};

export const RecSysInput = ({
  className,
  value,
  onChange,
  onSearchResults,
}) => {
  return (
    <>
      <RecAutocomplete
        value={value}
        onChange={onChange}
        onSearchResults={onSearchResults}
        className={className}
      />
    </>
  );
};

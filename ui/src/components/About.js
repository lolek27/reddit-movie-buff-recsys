export function About() {
  const goToGithub = () => {
    window.open(
      "https://github.com/lolek27/reddit-movie-buff/blob/main/README.md",
      "_blank"
    );
  };
  return (
    <div className="about" onClick={goToGithub()}>
      About Reddit Movie Buff
    </div>
  );
}

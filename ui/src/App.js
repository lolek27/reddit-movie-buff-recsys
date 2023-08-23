import { ParallaxProvider } from 'react-scroll-parallax';
import { RecSys } from './RecsSys';


function App() {

  return (
    <ParallaxProvider>
    <div className="App">
      <div className="app-container">
       <RecSys />
      </div>
    </div>
    </ParallaxProvider>
  );
}

export default App;

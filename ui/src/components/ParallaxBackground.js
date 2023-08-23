
import React from 'react';
import avatar from '../assets/avatar.png';
import luke from '../assets/luke1.png';
import lala from '../assets/lala1.png';
import interstellar from '../assets/interstellar.png';
import br2 from '../assets/bladerunner2.png';
import kb from '../assets/killbill.png';
import iron from '../assets/ironman.png';
import hobbit from '../assets/hobbit.png';
import alien from '../assets/alien1.png';
import neo from '../assets/neo.png';
import mia from '../assets/mia1.png';
import avengers from '../assets/avengers.png';
import deathstar from '../assets/deathstar.png';
import melancholy from '../assets/melancholy.png';
import meryl from '../assets/meryl.png';
import jurassic from '../assets/jurassic.png';
import bladerunner from '../assets/blade2.png';
import eye from '../assets/eye.png';
import sarah from '../assets/sarah1.png';
import inception from '../assets/inception.png';
import pursuit from '../assets/pursuit.png';
import xwing from '../assets/xwing.png';
import proposal from '../assets/proposal.png';
import { Parallax } from 'react-scroll-parallax';


export function ParallaxBackground () {
    const parConfigs = [{
         translateX: ['-20%', '70%'], 
         translateY: ['-120%', '70%'],
         opacity: ['0.5', '1.0', 'easeInCubic']
        },
        {
            translateY: ["250%", "-50%", "easeInOutQuad"],
             translateX: ['100%', '-50%'],
              opacity: ['0.1', '1.0', 'easeInCubic'],
            scaleX: ['0', '2', 'easeIn'],
            scaleY: ['0', '2', 'easeIn']
        },
        {
            scale: ['0', '2.5'],
            translateX: ["100%", "-100%"],
            opacity: ['0.5', '1.0', 'easeInCubic'],
        },
        {
            translateX: ['0%', '60%'],
            translateY: ['100%', '-100%'],
            opacity: ['0.5', '0.8', 'easeInCubic'],
        },
        {
            scale: ["0", "2"],
            translateX: ['-100%', '100%'],
            opacity: ['0.5', '0.9', 'easeInCubic'],
        },
        {
            translateX: ['0%', '-60%'],
            translateY: ['100%', '-100%'],
            opacity: ['0.5', '1.0', 'easeInCubic'],
        },
        {
            translateX: ['-40%', '-45%'],
            translateY: ['60%', '-10%'],
            opacity: ['0.8', '0.0', 'easeInCubic'],
        },
        {
            translateX: ['150%', '-50%'], 
            opacity: ['0.5', '1.0', 'easeInCubic'],
            translateY: ['-100%', '-10%'],
        },
        {
        translateX: ['-30%', '-50%'], 
        opacity: ['0.9', '0.4', 'easeInCubic'],
        translateY: ['-100%', '-10%'],
        },
        {
            translateX: ['20%', '55%'],
            translateY: ['170%', '-20%'],
            opacity: ['0.8', '0.0', 'easeInCubic'],
        },
        {
            translateX: ['-40%', '70%'], 
            opacity: ['0.5', '1.0', 'easeInCubic']
           },
    ];

    const ps = [ 
        [<img src={luke}  className='img1' alt='movie' />, parConfigs[0], 'img-start'],
        [<img src={avatar}  className='img0' alt='movie' />, parConfigs[1], 'img-start'],
        [<img src={interstellar}  className='img0' alt='movie' />, parConfigs[7], 'img-start'],
        [<img src={lala}  className='img1' alt='movie' />, parConfigs[8], 'img-start'],
        [<img src={br2}  className='img-mega-full' alt='movie' />, parConfigs[2], 'img-full'],
        [<img src={kb}  className='img1' alt='movie' />, parConfigs[3], ''],
        [<img src={iron}  className='img0' alt='movie' />, parConfigs[4], ''],
        [<img src={alien}  className='img1' alt='movie' />, parConfigs[5], ''],
        [<img src={hobbit}  className='img1 mb-3' alt='movie' />, parConfigs[10], ''],
        [<img src={deathstar}  className='img0 img-full' alt='movie' />, parConfigs[2], 'img-full'],
        [<img src={melancholy}  className='img0' alt='movie' />, parConfigs[6], ''],
        [<img src={neo}  className='img0' alt='movie' />, parConfigs[9], 'img-full'],
        [<img src={avengers}  className='img2' alt='movie' />, parConfigs[3], 'img-full'],
        [<img src={meryl}  className='img0' alt='movie' />, parConfigs[6], ''],
        [<img src={mia}  className='img0 img-full' alt='movie' />, parConfigs[4], ''],
        [<img src={jurassic}  className='img1 mt-5' alt='movie' />, parConfigs[2], 'img-full'],
        [<img src={bladerunner}  className='img2 mt-5' alt='movie' />, parConfigs[9], 'img-full'],
        [<img src={eye}  className='img1' alt='movie' />, parConfigs[6], ''],
        [<img src={sarah}  className='img0 mb-5' alt='movie' />, parConfigs[9], 'img-full'],
        [<img src={inception}  className='img1 mt-5' alt='movie' />, parConfigs[2], 'img-full'],
        [<img src={pursuit}  className='img1' alt='movie' />, parConfigs[6], ''],
        [<img src={xwing}  className='img0' alt='movie' />, parConfigs[4], ''],
        [<img src={proposal}  className='img0' alt='movie' />, parConfigs[9], 'img-full'],
    ]

    return (<div className="parallax-background">
         {ps.map((p, i) => {
          return (
            <div key={i} style={{ perspective: 400 }} className={`p-item ${p[2]}`}>
              <Parallax {...p[1]}>
                {p[0]}
              </Parallax>
            </div>
          );
        })}
       
    </div>);
}
  
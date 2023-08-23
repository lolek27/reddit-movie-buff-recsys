import React from 'react';
import Skeleton, {SkeletonTheme} from 'react-loading-skeleton';
import 'react-loading-skeleton/dist/skeleton.css'

export const LoadingSkeleton = ({ className }) => {
    return (
        <SkeletonTheme baseColor="#1313134d" highlightColor="#f54b002e">
        <div className={`movie-card + ${className}`}>
            <div className='body' style={{backgroundColor: '#131313'}}>
                <Skeleton width={200} height={300}/>
            <div className='content'  style= {{width: '70%'}}>
                <h3 className='title'>
                    <Skeleton count={1} width={200} height={30}/>
                </h3>
                <div className='subtitle'>
                    <Skeleton count={1} width={280} height={'20px'} />
                </div>
                <div className='tagline'><Skeleton count={1} width={350} /></div>
                <div className='overview' style= {{width: '100%'}}>
                  <div><Skeleton count={1} width={100} height={'20px'} /></div>
                  <p ><Skeleton count={3}/></p>
                </div>
                <div className='db-buttons'>
                    <Skeleton height={42} width={150} />
                    <Skeleton height={42} width={150} />
                </div>
            </div>
        </div>
        </div>
        </SkeletonTheme>
    );
}

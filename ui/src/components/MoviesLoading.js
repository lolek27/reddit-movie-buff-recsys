import { LoadingSkeleton } from './LoadingSkeleton'


const LOAD_SIZE = 7

export function MoviesLoading() {
    return <>{Array.from(Array(LOAD_SIZE)).map(() => (<LoadingSkeleton className='mb-4'/>))}</>
}
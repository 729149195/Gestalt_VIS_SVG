import { resetSubmissionCount } from './src/api/counter.js';

console.log('Resetting submission count...');
resetSubmissionCount()
  .then(() => {
    console.log('Count reset successful');
    process.exit(0);
  })
  .catch(error => {
    console.error('Error resetting count:', error);
    process.exit(1);
  }); 
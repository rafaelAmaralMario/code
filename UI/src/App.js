import React from 'react';
import MainPage from './components/MainPage';
import TrainingPage from './components/TrainingPage';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

const theme = {}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route exact path="/" element={<MainPage />} />

          <Route path="/training" element={<TrainingPage />}/>
            
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;

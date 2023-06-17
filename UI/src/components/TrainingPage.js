import React, { useState } from 'react';
import axios from 'axios';
import { Typography, Box, Button, LinearProgress, Container } from '@mui/material';

function TrainingPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [name, setName] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);


  const handleInputChange = (event) => {
    setName(event.target.value);
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleFileUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('input_data', selectedFile);
    formData.append('name', name);

    try {
      const response = await axios.post('http://localhost:5000/train_unsupervised', formData, {
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });

      console.log('Training completed:', response.data);
    } catch (error) {
      console.error('Error training:', error);
    }
  };

  return (
    <Container maxWidth="sm">
      <Box my={4}>
        <Typography variant="h3" component="h1" align="center">
          Training Page
        </Typography>

        <TextField
          label="Name of training."
          variant="outlined"
          fullWidth
          margin="normal"
          value={name}
          onChange={handleInputChange}
        />
        <Box my={2}>
          <input type="file" onChange={handleFileChange} />
        </Box>
        
        <Box my={2}>
          <Button variant="contained" color="primary" onClick={handleFileUpload}>
            Start Training
          </Button>
        </Box>
        
        {uploadProgress > 0 && (
          <Box my={2}>
            <LinearProgress variant="determinate" value={uploadProgress} />
            <Typography variant="body1" align="center">
              Training Progress: {uploadProgress}%
            </Typography>
          </Box>
        )}
      </Box>
    </Container>
  );
}

export default TrainingPage;

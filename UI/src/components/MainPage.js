import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Typography, Box, TextField, Button, Container } from '@mui/material';

function MainPage() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');

  useEffect(() => {
    fetchMessages();
  }, []);

  const fetchMessages = async () => {
    try {
      const response = await axios.get('http://localhost:5000/message');
      setMessages(response.history);
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const sendMessage = async () => {
    if (inputText.trim() === '') return;

    try {
      const response = await axios.post('http://localhost:5000/message', {
        message: inputText
      });

      setMessages([...messages, response.data]);
      setInputText('');
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  return (
    <Container maxWidth="sm">
      <Box my={4}>
        <Typography variant="h3" component="h1" align="center">
          Welcome to the Main Page!
        </Typography>
        <Typography variant="body1" align="center" paragraph>
          This is the main page of the application.
        </Typography>

        <Typography variant="h4" component="h2">
          Chat:
        </Typography>
        <div className="messages">
          {messages.map((message) => (
            <div key={message.id} className="message">
              <span className="sender">{message.sender}: </span>
              <span className="text">{message.text}</span>
            </div>
          ))}
        </div>

        <TextField
          label="Type your message..."
          variant="outlined"
          fullWidth
          value={inputText}
          onChange={handleInputChange}
          margin="normal"
        />
        <Button variant="contained" color="primary" onClick={sendMessage}>
          Send
        </Button>
      </Box>
    </Container>
  );
}

export default MainPage;

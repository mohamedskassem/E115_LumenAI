document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const sqlDisplay = document.getElementById('sqlDisplay');
    const sqlCode = document.getElementById('sqlCode');
    const modelSelector = document.getElementById('modelSelector');

    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent default Enter behavior (new line)
            sendMessage();
        }
    });
    userInput.addEventListener('input', autoGrowTextarea);

    // --- Functions ---
    function autoGrowTextarea() {
        userInput.style.height = 'auto'; // Reset height
        userInput.style.height = userInput.scrollHeight + 'px'; // Set to scroll height
        // Enable/disable send button based on input
        sendButton.disabled = userInput.value.trim() === '';
    }

    function addMessage(text, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');

        if (sender === 'user') {
            messageDiv.classList.add('user-message');
        } else if (isError) {
            messageDiv.classList.add('error-message');
            text = `<strong>Error:</strong> ${text}`;
        } else {
            messageDiv.classList.add('bot-message');
        }

        // Basic Markdown-like formatting for newlines
        const formattedText = text.replace(/\n/g, '<br>');
        messageDiv.innerHTML = `<p>${formattedText}</p>`; // Use innerHTML to render <br>
        chatbox.appendChild(messageDiv);

        // Scroll to the bottom
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function toggleLoading(show) {
        loadingIndicator.style.display = show ? 'flex' : 'none';
        sendButton.disabled = show;
        userInput.disabled = show;
    }

    function displaySql(sql) {
        if (sql) {
            sqlCode.textContent = sql;
            sqlDisplay.style.display = 'block';
        } else {
            sqlDisplay.style.display = 'none';
        }
         // Scroll chatbox down after potentially showing SQL
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    async function sendMessage() {
        const question = userInput.value.trim();
        if (question === '') return;

        addMessage(question, 'user');
        userInput.value = '';
        autoGrowTextarea(); // Reset textarea height after sending
        toggleLoading(true);
        displaySql(null); // Hide previous SQL

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    question: question,
                    model: modelSelector.value 
                }),
            });

            toggleLoading(false);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response.'}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Display SQL if provided
            displaySql(data.sql_query);

            // Add bot response
            if (data.message) {
                addMessage(data.message, 'bot');
            } else if (data.error) { // Handle errors returned in the JSON body gracefully
                 addMessage(data.error, 'bot', true);
            } else {
                 addMessage("Received an empty response from the server.", 'bot', true);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            toggleLoading(false);
            addMessage(error.message || "Could not connect to the server or an unknown error occurred.", 'bot', true);
            displaySql(null); // Ensure SQL display is hidden on error
        }
    }

    // Initial setup
    autoGrowTextarea(); // Adjust initial height and button state
}); 
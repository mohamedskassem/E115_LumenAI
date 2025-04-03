document.addEventListener('DOMContentLoaded', () => {
    // UI Sections
    const dataLoaderSection = document.getElementById('dataLoaderSection');
    const chatSection = document.getElementById('chatSection');

    // Data Loader Elements
    const csvUpload = document.getElementById('csvUpload');
    const fileInfo = document.getElementById('fileInfo');
    const uploadButton = document.getElementById('uploadButton');
    const uploadStatus = document.getElementById('uploadStatus');

    // Chat Section Elements
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const sqlDisplay = document.getElementById('sqlDisplay');
    const sqlCode = document.getElementById('sqlCode');
    const modelSelector = document.getElementById('modelSelector');
    const refreshButton = document.getElementById('refreshIndexButton');
    const removeDataButton = document.getElementById('removeDataButton');

    // Initial State Check
    checkInitialStatus();

    // --- Event Listeners ---
    // Data Loader
    csvUpload.addEventListener('change', handleFileSelection);
    uploadButton.addEventListener('click', uploadFiles);
    // Chat
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent default Enter behavior (new line)
            sendMessage();
        }
    });
    userInput.addEventListener('input', autoGrowTextarea);
    refreshButton.addEventListener('click', refreshIndex);
    removeDataButton.addEventListener('click', removeData);

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

    // --- Refresh Index Function ---
    async function refreshIndex() {
        addMessage("Attempting to refresh index... This may take a moment.", 'system'); // Use a distinct style? For now, 'system' uses bot style
        toggleLoading(true); // Disable input while refreshing
        refreshButton.disabled = true;

        try {
            const response = await fetch('/refresh-index', {
                method: 'POST'
            });

            toggleLoading(false);
            refreshButton.disabled = false;

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            addMessage(data.message || "Index refreshed successfully!", 'system');

        } catch (error) {
            console.error('Error refreshing index:', error);
            toggleLoading(false);
            refreshButton.disabled = false;
            addMessage(error.message || "Could not refresh index.", 'bot', true); // Show error
        }
    }

    // --- UI State Management ---
    function showDataLoader() {
        dataLoaderSection.style.display = 'flex';
        chatSection.style.display = 'none';
        clearChat(); // Clear chat when showing loader
    }

    function showChatInterface() {
        dataLoaderSection.style.display = 'none';
        chatSection.style.display = 'flex';
        addMessage("Data loaded. Ask me anything about it!", 'system');
    }

    function clearChat() {
        chatbox.innerHTML = ''; // Clear all messages
    }

    // --- Initial Status Check ---
    async function checkInitialStatus() {
        try {
            const response = await fetch('/status');
            if (!response.ok) throw new Error('Failed to fetch status');
            const data = await response.json();
            if (data.is_ready) {
                showChatInterface();
            } else {
                showDataLoader();
            }
        } catch (error) {
            console.error("Error checking initial status:", error);
            showDataLoader(); // Default to loader on error
            // Optionally show an error message to the user
            uploadStatus.textContent = "Error contacting server. Please refresh.";
            uploadStatus.className = 'status-message error';
        }
    }

    // --- Data Loader Functions ---
    function handleFileSelection() {
        const files = csvUpload.files;
        if (files.length > 0) {
            fileInfo.textContent = `${files.length} file(s) selected`;
            uploadButton.disabled = false;
        } else {
            fileInfo.textContent = "No files selected";
            uploadButton.disabled = true;
        }
    }

    async function uploadFiles() {
        const files = csvUpload.files;
        if (files.length === 0) return;

        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        uploadStatus.textContent = "Uploading and processing files... This may take time.";
        uploadStatus.className = 'status-message';
        uploadButton.disabled = true;

        try {
            const response = await fetch('/upload-data', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            uploadStatus.textContent = data.message || "Upload successful! Initializing agent...";
            uploadStatus.className = 'status-message success';
            // Reset file input
            csvUpload.value = ''; 
            handleFileSelection();
            // Switch UI
            showChatInterface(); 

        } catch (error) {
            console.error('Error uploading files:', error);
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.className = 'status-message error';
            uploadButton.disabled = false; // Re-enable on error
        }
    }

    // --- Remove Data Function ---
    async function removeData() {
        if (!confirm("Are you sure you want to remove the current data and index? This cannot be undone.")) {
            return;
        }
        
        addMessage("Removing data and index...", 'system');
        toggleLoading(true); // Use loading indicator logic
        removeDataButton.disabled = true;

        try {
            const response = await fetch('/remove-data', {
                method: 'POST'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || data.message || `HTTP error! status: ${response.status}`);
            }

            uploadStatus.textContent = data.message || "Data removed successfully."; // Show status in upload section
            uploadStatus.className = 'status-message success';
            showDataLoader(); // Switch back to data loader UI

        } catch (error) {
            console.error('Error removing data:', error);
            addMessage(`Error removing data: ${error.message}`, 'bot', true); // Show error in chat
        } finally {
             toggleLoading(false);
             removeDataButton.disabled = false;
        }
    }
}); 
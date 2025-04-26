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
    const modelSelector = document.getElementById('modelSelector');
    const removeDataButton = document.getElementById('removeDataButton');

    // New elements for multi-chat
    const chatList = document.getElementById('chatList');
    const newChatButton = document.getElementById('newChatButton');

    // State variables
    let currentChatId = null;
    let chatHistories = {}; // Store message history per chat { chatId: [messages...]} 
    let chatDisplayNames = {}; // Store display names { chatId: displayName }

    // Initial State Check
    checkStatusAndLoadChats();

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
    removeDataButton.addEventListener('click', removeData);

    // --- Functions ---
    function autoGrowTextarea() {
        userInput.style.height = 'auto'; // Reset height
        userInput.style.height = userInput.scrollHeight + 'px'; // Set to scroll height
        // Enable/disable send button based on input
        sendButton.disabled = userInput.value.trim() === '';
    }

    function toggleLoading(show) {
        loadingIndicator.style.display = show ? 'flex' : 'none';
        sendButton.disabled = show;
        userInput.disabled = show;
    }

    async function sendMessage() {
        const question = userInput.value.trim();
        const selectedModel = modelSelector.value;

        if (!question) return;
        if (!currentChatId) {
             addMessageToChatbox('system', 'Please select or start a chat first.', 'error', null, false);
             return;
        }

        addMessageToChatbox('user', question, 'normal', null, true);
        userInput.value = '';
        autoGrowTextarea(); // Reset textarea height after sending
        toggleLoading(true);

        // Check if we need to generate a title for this chat
        console.log(`Checking title for [${currentChatId}]: current display name is '${chatDisplayNames[currentChatId]}'`); // Debug log
        const needsTitle = chatDisplayNames[currentChatId] === "New Chat";

        // --- Debugging Log ---
        const requestBody = { 
            question: question, 
            model: selectedModel, 
            chat_id: currentChatId, 
            generate_title: needsTitle
        };
        console.log('Sending /query with:', requestBody);
        // --- End Debugging Log ---

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody), // Use the logged body
            });

            toggleLoading(false);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response.'}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Add bot response using addMessageToChatbox to save history
            const messageType = data.error ? 'error' : 'normal';
            const messageText = data.message || data.error || "Received an empty response from the server.";

            addMessageToChatbox('bot', messageText, messageType, data.sql_query, true);

            // --- Update Chat Title in UI if Changed ---
            const receivedChatId = currentChatId; // Chat ID for which response was received
            const receivedTitle = data.chat_title;

            // Check if the title actually changed AND the chat still exists locally
            if (receivedTitle && chatDisplayNames[receivedChatId] && chatDisplayNames[receivedChatId] !== receivedTitle) {
                console.log(`Updating title for chat ${receivedChatId} from '${chatDisplayNames[receivedChatId]}' to '${receivedTitle}'`);
                chatDisplayNames[receivedChatId] = receivedTitle; // Update local store

                // Re-render the chat list using the updated display names
                const chatListArray = Object.keys(chatDisplayNames).map(id => ({
                    chat_id: id,
                    title: chatDisplayNames[id]
                }));
                renderChatList(chatListArray);
                // highlightSelectedChat(currentChatId); // Re-highlighting is handled by renderChatList now based on currentChatId
            }
            // --- End Title Update ---

        } catch (error) {
            console.error('Error sending message:', error);
            toggleLoading(false);
            addMessageToChatbox('bot', error.message || "Could not connect to the server or an unknown error occurred.", 'error', null, false);
        }
    }

    // Initial setup
    autoGrowTextarea(); // Adjust initial height and button state

    // --- UI State Management ---
    function showDataLoader() {
        dataLoaderSection.style.display = 'flex';
        chatSection.style.display = 'none';
        clearChat(); // Clear chat when showing loader
    }

    function showChatInterface() {
        dataLoaderSection.style.display = 'none';
        chatSection.style.display = 'flex';
        addMessageToChatbox('system', "Data loaded. Ask me anything about it!", 'info', null, false);
    }

    function clearChat() {
        chatbox.innerHTML = ''; // Clear all messages
    }

    // --- Initial Status Check ---
    async function checkStatusAndLoadChats() {
        try {
            const response = await fetch('/status');
            if (!response.ok) throw new Error('Failed to fetch status');
            const data = await response.json();
            if (data.is_data_loaded) {
                showChatInterface();
                // Update local display names from server status
                chatDisplayNames = data.active_chats.reduce((acc, chat) => {
                    acc[chat.chat_id] = chat.title;
                    return acc;
                }, {});
                renderChatList(data.active_chats); // Pass full chat info
                const chatIds = data.active_chats.map(c => c.chat_id);

                if (chatIds.length > 0) {
                    // Select the first chat if none is selected or current is invalid
                    if (!currentChatId || !chatIds.includes(currentChatId)) {
                        selectChat(chatIds[0]);
                    } else {
                        // Re-select current to ensure UI is up-to-date
                        selectChat(currentChatId);
                    }
                    enableChatControls();
                } else {
                    // Data loaded but no chats exist (e.g., after delete all)
                    currentChatId = null;
                    chatbox.innerHTML = '<div class="message bot">Please start a new chat to begin.</div>';
                    disableChatControls();
                    newChatButton.disabled = false;
                    chatHistories = {}; // Clear chat histories as well
                }
            } else {
                showDataLoader();
                disableChatControls();
                chatList.innerHTML = ''; // Clear chat list
                currentChatId = null;
                chatDisplayNames = {}; // Clear display names
                chatHistories = {}; // Clear chat histories as well
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
            const fileNames = Array.from(files).map(f => f.name).join(', ');
            fileInfo.textContent = `${files.length} file(s) selected: ${fileNames}`;
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
            checkStatusAndLoadChats(); 

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
        
        addMessageToChatbox('system', "Removing data and index...", 'info', null, false);
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
            addMessageToChatbox('bot', `Error removing data: ${error.message}`, 'error', null, false);
        } finally {
             toggleLoading(false);
             removeDataButton.disabled = false;
        }
    }

    // --- Chat List Management ---
    function renderChatList(chats) { // Expects array of {chat_id, title}
        chatList.innerHTML = ''; // Clear existing list
        if (!chats || chats.length === 0) {
            const noChatsItem = document.createElement('li');
            noChatsItem.textContent = 'No active chats.';
            noChatsItem.style.fontStyle = 'italic';
            noChatsItem.style.color = '#6c757d';
            chatList.appendChild(noChatsItem);
            return;
        }

        chats.forEach(chat => {
            const chatId = chat.chat_id;
            // Use title from chat object (synced from server)
            const displayName = chat.title || `Chat ${chatId.substring(0, 8)}...`;
            const listItem = document.createElement('li');
            listItem.dataset.chatId = chatId; // Ensure the LI has the ID for highlightSelectedChat

            // Add chat title span
            const titleSpan = document.createElement('span');
            titleSpan.classList.add('chat-title');
            titleSpan.textContent = displayName;
            listItem.appendChild(titleSpan);

            // Attach select listener to the title span
            titleSpan.addEventListener('click', () => {
                selectChat(chatId);
            });

            // Add delete icon
            const deleteIcon = document.createElement('img'); // Create an img element
            deleteIcon.src = "static/delete.svg"; // Set the source to your SVG file
            deleteIcon.alt = "Delete Chat"; // Add alt text for accessibility
            deleteIcon.classList.add('delete-chat-icon');
            deleteIcon.title = 'Delete Chat';
            deleteIcon.dataset.chatId = chatId; // Assign chatId for the handler
            deleteIcon.addEventListener('click', (event) => {
                event.stopPropagation(); // Prevent selectChat from firing
                handleDeleteChat(chatId); // Call delete with specific ID
            });
            listItem.appendChild(deleteIcon);
            if (chatId === currentChatId) {
                listItem.classList.add('selected');
            }
            chatList.appendChild(listItem);
        });
        // Delete icons handle their own state
    }

    function selectChat(chatId) {
        if (currentChatId === chatId && chatbox.children.length > 0 && chatHistories[chatId]?.length > 0) {
            highlightSelectedChat(chatId);
            return;
        }

        console.log(`Selecting chat: ${chatId}`);
        currentChatId = chatId;
        highlightSelectedChat(chatId);

        chatbox.innerHTML = ''; // Clear current messages

        const history = chatHistories[chatId] || [];

        history.forEach(msg => {
             // Add message without saving it again to history
             addMessageToChatbox(msg.sender, msg.text, msg.type, msg.sql, false); 
        });

        if (history.length === 0) {
            const title = chatDisplayNames[chatId] || `Chat ${chatId.substring(0, 8)}...`;
            addMessageToChatbox('bot', `Selected ${title}. Ask a question about the loaded data.`, 'info', null, false);
        }
        
        enableChatControls(); 
        userInput.focus();
        // Delay scroll slightly to allow rendering
        setTimeout(() => { chatbox.scrollTop = chatbox.scrollHeight; }, 0); 
    }

    function highlightSelectedChat(chatId) {
        const items = chatList.querySelectorAll('li');
        items.forEach(item => {
            if (item.dataset.chatId === chatId) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
        // Delete icons are managed individually now
    }

    function handleNewChat() {
        console.log('Requesting new chat...');
        newChatButton.disabled = true; 

        fetch('/chats', { method: 'POST' })
            .then(response => {
                if (!response.ok) {
                    // Try parsing error message from backend
                    return response.json().then(errData => {
                         throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                    }).catch(() => {
                         // Fallback if parsing error fails
                         throw new Error(`HTTP error! status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.chat_id) {
                    const newChatId = data.chat_id;
                    console.log('New chat created:', newChatId);
                    // --- Direct UI Update ---
                    // 1. Add to local history store & display names
                    chatHistories[newChatId] = []; // Initialize empty history
                    chatDisplayNames[newChatId] = "New Chat"; // Initialize display name

                    // 2. Re-render the chat list with the new chat
                    const chatListArray = Object.keys(chatDisplayNames).map(id => ({
                        chat_id: id,
                        title: chatDisplayNames[id]
                    }));
                    renderChatList(chatListArray);

                    // 3. Explicitly select the new chat
                    selectChat(newChatId); 
                    // --- End Direct UI Update ---

                    // checkStatusAndLoadChats(); // No longer needed here for immediate update
                } else {
                    throw new Error(data.error || 'Failed to create chat, no ID returned.');
                }
            })
            .catch(error => {
                console.error('Error creating new chat:', error);
                addMessageToChatbox('system', `Error creating new chat: ${error.message}`, 'error', null, false);
            })
            .finally(() => {
                 // Re-enable button (handled within selectChat -> enableChatControls)
                 // newChatButton.disabled = false; // Let enableChatControls handle this
            });
    }

    function handleDeleteChat(chatIdToDelete) { // Accept chatId as argument
        if (!chatIdToDelete) {
            console.error("handleDeleteChat called without a chatId");
            return;
        }
        const chatCount = Object.keys(chatDisplayNames).length;
        if (chatCount <= 1) {
             addMessageToChatbox('system', 'Cannot delete the last remaining chat.', 'error', null, false);
             return;
        }

        const chatTitle = chatDisplayNames[chatIdToDelete] || `Chat ${chatIdToDelete.substring(0, 8)}...`;
        if (!confirm(`Are you sure you want to delete '${chatTitle}'?`)) {
            return;
        }

        console.log(`Requesting delete for chat: ${chatIdToDelete}`);
        // Disable the specific icon? Maybe not needed, list will refresh.

        fetch(`/chats/${chatIdToDelete}`, { method: 'DELETE' })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errData => {
                        throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                    }).catch(() => {
                         throw new Error(`HTTP error! status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Chat deleted:', data.message);
                // Remove from local stores (should already be gone from server)
                delete chatHistories[chatIdToDelete];
                delete chatDisplayNames[chatIdToDelete]; // Remove display name
                // If the deleted chat was the current one, clear selection
                if (currentChatId === chatIdToDelete) {
                    currentChatId = null;
                    chatbox.innerHTML = '';
                    disableChatControls(); // Go to a neutral state before reloading
                }
                checkStatusAndLoadChats(); // Refresh status and chat list
            })
            .catch(error => {
                console.error('Error deleting chat:', error);
                addMessageToChatbox('system', `Error deleting chat: ${error.message}`, 'error', null, false);
                // Re-enable icon? Handled by list refresh.
            });
    }

    // --- Message Handling ---
    function addMessageToChatbox(sender, text, type = 'normal', sql = null, saveToHistory = true) { 

        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        if (sender === 'user') {
            messageElement.classList.add('user');
        } else {
            messageElement.classList.add('bot'); 
        }

        if (type === 'error') {
            messageElement.classList.add('error');
            messageElement.innerHTML = `<strong>Error:</strong> `; 
            messageElement.appendChild(document.createTextNode(text)); 
        } else if (type === 'info' || sender === 'system') {
             messageElement.classList.add('info'); 
             messageElement.textContent = text;
        } else { // Handle 'normal' type messages
            if (sender === 'bot') {
                try {
                     messageElement.innerHTML = marked.parse(text, { breaks: true });
                 } catch (e) {
                     console.error("Markdown parsing error:", e);
                     messageElement.textContent = text; 
                 }
            } else { // Sender is 'user' 
                messageElement.textContent = text; // Set text content directly
            } 
        }

        chatbox.appendChild(messageElement);

        // Adjust scrolling behavior
        // Only auto-scroll if near the bottom or if loading history
        const shouldScroll = (chatbox.scrollHeight - chatbox.scrollTop - chatbox.clientHeight < 100) || !saveToHistory;
        if (shouldScroll) {
             // Use requestAnimationFrame for smoother scrolling after render
             requestAnimationFrame(() => {
                 chatbox.scrollTop = chatbox.scrollHeight;
             });
        }

        // --- NEW: Append SQL toggle if applicable --- 
        // Append only for NEW bot messages of type 'normal' that HAVE SQL
        if (sender === 'bot' && type === 'normal' && sql && saveToHistory) {
            const sqlToggleDiv = document.createElement('div');
            sqlToggleDiv.classList.add('sql-toggle');
            sqlToggleDiv.innerHTML = `
                <details>
                    <summary>Show Generated SQL</summary>
                    <pre><code>${escapeHtml(sql)}</code></pre>
                </details>
            `;
            messageElement.appendChild(sqlToggleDiv);

            // Scroll down after adding toggle, as it increases message height
            requestAnimationFrame(() => {
                 chatbox.scrollTop = chatbox.scrollHeight;
             });
        }

        // --- History Saving Logic --- 
        // Restore original condition
        if (saveToHistory && currentChatId && type !== 'info' && sender !== 'system') { 
            if (!chatHistories[currentChatId]) {
                chatHistories[currentChatId] = [];
            }
            const history = chatHistories[currentChatId];
            // Use the correct message text based on type
            const textToSave = (type === 'error') ? `Error: ${text}` : text; 
            const messageToSave = { sender, text: textToSave, type, sql }; // Prepare object

            // Save unconditionally if the main 'if' condition passed
            history.push(messageToSave);
        }
    }

    function enableChatControls() {
        userInput.disabled = false;
        sendButton.disabled = userInput.value.trim() === ''; // Disable send if input empty
        modelSelector.disabled = false;
        removeDataButton.disabled = false;
        newChatButton.disabled = false;
        // Delete icons are handled individually
    }

    function disableChatControls() {
        userInput.disabled = true;
        sendButton.disabled = true;
        modelSelector.disabled = true;
        removeDataButton.disabled = true; 
        newChatButton.disabled = true;
        userInput.value = '';
        adjustTextareaHeight();
    }

    function adjustTextareaHeight() {
        userInput.style.height = 'auto'; 
        const scrollHeight = userInput.scrollHeight;
        const maxHeight = 100; 
        userInput.style.height = Math.min(scrollHeight, maxHeight) + 'px';
        userInput.style.overflowY = scrollHeight > maxHeight ? 'auto' : 'hidden';
        // Also update send button state based on input
        if (currentChatId) { // Only enable if a chat is active
             sendButton.disabled = userInput.value.trim() === '';
        } else {
             sendButton.disabled = true; // Ensure disabled if no chat selected
        }
    }

    // Helper function to escape HTML for displaying in code blocks
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    // --- Event Listeners for New Buttons ---
    newChatButton.addEventListener('click', handleNewChat);
    // Delete listeners are added dynamically in renderChatList

    // --- Initial Load ---
    checkStatusAndLoadChats();
    adjustTextareaHeight();

}); 
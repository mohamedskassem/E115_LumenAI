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
    let chatHistories = {}; // Store message history per chat { chatId: [messages...] }
    let chatDisplayNames = {}; // Store display names { chatId: displayName }
    let isFirstMessage = {}; // Track if the next message is the first for title gen {chatId: true/false}

    // Initial State Check
    // Use sessionStorage to persist chat ID across refreshes within the same tab
    initializeChat();
    // checkStatusAndLoadChats(); // Called by initializeChat now

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
             addMessageToChatbox('system', 'Error: No active chat selected. Please start a new chat.', 'error', null, false);
             console.error("SendMessage called with no currentChatId");
             return;
        }

        // Determine if title generation is needed
        const generateTitle = !!isFirstMessage[currentChatId]; // True if flag is set
        console.log(`Sending message for chat ${currentChatId}. Generate title: ${generateTitle}`);

        addMessageToChatbox('user', question, 'normal', null, true);
        userInput.value = '';
        autoGrowTextarea(); // Reset textarea height after sending
        toggleLoading(true);

        const requestBody = {
            question: question,
            model: selectedModel,
            chat_id: currentChatId,
            generate_title: generateTitle
        };
        console.log('Sending /query with:', requestBody);

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
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
            const receivedChatId = currentChatId;
            const receivedTitle = data.chat_title;

            if (receivedTitle && chatDisplayNames[receivedChatId] && chatDisplayNames[receivedChatId] !== receivedTitle) {
                console.log(`Updating title for chat ${receivedChatId} from '${chatDisplayNames[receivedChatId]}' to '${receivedTitle}'`);
                chatDisplayNames[receivedChatId] = receivedTitle; // Update local store

                const chatListArray = Object.keys(chatDisplayNames).map(id => ({
                    chat_id: id,
                    title: chatDisplayNames[id]
                }));
                renderChatList(chatListArray);
            }

            // Mark that the first message has been sent for this chat
            if (generateTitle) {
                isFirstMessage[currentChatId] = false;
                console.log(`Title generation flag set to false for chat ${currentChatId}`);
            }

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

            uploadStatus.textContent = data.message || "Upload successful! Initializing...";
            uploadStatus.className = 'status-message success';
            csvUpload.value = '';
            handleFileSelection();

            // Clear existing session state and initialize fresh after upload
            currentChatId = null;
            sessionStorage.removeItem('currentChatId');
            chatHistories = {};
            chatDisplayNames = {};
            isFirstMessage = {};
            await initializeChat(); // Re-initialize after successful upload

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

            // Clear state and switch UI
            currentChatId = null;
            sessionStorage.removeItem('currentChatId');
            chatHistories = {};
            chatDisplayNames = {};
            isFirstMessage = {};

        } catch (error) {
            console.error('Error removing data:', error);
            addMessageToChatbox('bot', `Error removing data: ${error.message}`, 'error', null, false);
        } finally {
             toggleLoading(false);
             removeDataButton.disabled = true; // Technically disabled by disableChatControls
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
        // Avoid redundant processing if already selected
        console.log(`[selectChat] Attempting to select chat: ${chatId}. Current: ${currentChatId}`);
        if (currentChatId === chatId && chatbox.children.length > 0 && (chatHistories[chatId]?.length > 0 || chatbox.querySelector('.message'))) {
             console.log(`[selectChat] Chat ${chatId} already selected and has content. Highlighting.`);
             highlightSelectedChat(chatId); // Ensure highlight is correct
             return;
        }

        console.log(`[selectChat] Selecting chat: ${chatId}`);
        currentChatId = chatId;
        sessionStorage.setItem('currentChatId', chatId); // Save selection
        console.log(`[selectChat] Set currentChatId to ${currentChatId} and saved to sessionStorage.`);
        highlightSelectedChat(chatId);

        chatbox.innerHTML = ''; // Clear current messages
        console.log(`[selectChat] Chatbox cleared for ${chatId}.`);

        // Retrieve history OR initialize if it doesn't exist locally yet
        const history = chatHistories[currentChatId] || [];
        console.log(`[selectChat] Retrieved history for ${currentChatId}. Length: ${history.length}`);
        if (!chatHistories[currentChatId]) {
             console.log(`[selectChat] Initializing local history array for ${currentChatId}.`);
             chatHistories[currentChatId] = history;
             // Assume it's a new chat if history is empty locally, mark for title gen
             // unless we already have a non-default display name
             if (!chatDisplayNames[currentChatId] || chatDisplayNames[currentChatId] === "New Chat") {
                 isFirstMessage[currentChatId] = true;
                 console.log(`[selectChat] Marking chat ${currentChatId} for title generation.`);
             }
        }

        console.log(`[selectChat] Rendering ${history.length} messages for ${currentChatId}...`);
        history.forEach(msg => {
            addMessageToChatbox(msg.sender, msg.text, msg.type, msg.sql, false);
        });
        console.log(`[selectChat] Finished rendering history for ${currentChatId}.`);

        if (history.length === 0) {
            const title = chatDisplayNames[currentChatId] || `Chat ${currentChatId.substring(0, 8)}...`;
            console.log(`[selectChat] Adding initial info message for empty chat ${currentChatId}.`);
            addMessageToChatbox('bot', `Selected ${title}. Ask a question.`, 'info', null, false);
        }

        enableChatControls();
        userInput.focus();
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
                delete isFirstMessage[chatIdToDelete];
                // If the deleted chat was the current one, clear selection
                if (currentChatId === chatIdToDelete) {
                    currentChatId = null;
                    sessionStorage.removeItem('currentChatId');
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

        // If this was the first *user* message, mark it so title gen flag isn't set next time
        if(sender === 'user' && isFirstMessage[currentChatId]) {
            console.log(`First user message saved for ${currentChatId}, setting isFirstMessage to false.`);
            isFirstMessage[currentChatId] = false;
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

    // NEW: Function to initialize or restore chat session
    async function initializeChat() {
        console.log("[initializeChat] START");
        const storedChatId = sessionStorage.getItem('currentChatId');
        console.log("[initializeChat] Stored chat ID:", storedChatId);

        // Fetch current server state regardless
        const serverStatus = await fetchServerStatus();
        console.log("[initializeChat] Server Status:", serverStatus);

        if (serverStatus && serverStatus.is_data_loaded) {
            // Update local maps with server data
            chatDisplayNames = serverStatus.active_chats.reduce((acc, chat) => {
                acc[chat.chat_id] = chat.title;
                return acc;
            }, {});
            renderChatList(serverStatus.active_chats); // Render full list

            const activeChatIds = serverStatus.active_chats.map(c => c.chat_id);

            // If a valid chat ID exists in sessionStorage for this tab, try to restore it
            if (storedChatId && activeChatIds.includes(storedChatId)) {
                console.log(`[initializeChat] Restoring session for chat ID: ${storedChatId}`);
                await selectChat(storedChatId); // Select the stored chat
                enableChatControls();
            } else {
                // If no valid ID in sessionStorage, ALWAYS create a new chat for this new session
                console.log("[initializeChat] No valid stored chat ID. Creating new chat.");
                if (storedChatId) { 
                    console.log("[initializeChat] Removing invalid stored ID:", storedChatId);
                    sessionStorage.removeItem('currentChatId'); 
                } // Clear invalid stored ID
                await createNewChat(); // Create a new one
                // enableChatControls() will be called by selectChat inside createNewChat
            }
        } else {
            // Data not loaded on server
            console.log("Data not loaded on server.");
            showDataLoader();
            disableChatControls();
            chatList.innerHTML = '';
            currentChatId = null;
            sessionStorage.removeItem('currentChatId');
            chatDisplayNames = {};
            chatHistories = {};
            isFirstMessage = {};
        }
    }

    // NEW: Function to explicitly create a new chat session
    async function createNewChat() {
        console.log('[createNewChat] Requesting new chat from server...');
        newChatButton.disabled = true;
        toggleLoading(true);
        let newChatId = null;

        try {
            const response = await fetch('/chats', { method: 'POST' });
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            if (data.chat_id) {
                newChatId = data.chat_id;
                console.log('[createNewChat] New chat created successfully:', newChatId);

                // Update local state immediately
                chatHistories[newChatId] = [];
                chatDisplayNames[newChatId] = "New Chat";
                isFirstMessage[newChatId] = true; // Mark for title generation
                console.log('[createNewChat] Local state updated for:', newChatId);

                // Re-render list and select
                const chatListArray = Object.keys(chatDisplayNames).map(id => ({
                    chat_id: id,
                    title: chatDisplayNames[id]
                }));
                console.log('[createNewChat] Rendering chat list:', chatListArray);
                renderChatList(chatListArray);
                console.log('[createNewChat] Calling selectChat for:', newChatId);
                await selectChat(newChatId); // Select the new chat
                addMessageToChatbox('system', "New chat started. Ask a question!", 'info', null, false);

            } else {
                throw new Error(data.error || 'Failed to create chat, no ID returned.');
            }
        } catch (error) {
            console.error('Error creating new chat:', error);
            // Display error in the chatbox if possible, otherwise fallback
            if (chatbox) {
                 addMessageToChatbox('system', `Error creating new chat: ${error.message}`, 'error', null, false);
            } else {
                 alert(`Error creating new chat: ${error.message}`);
            }
            // If creation fails, we might want to revert state or disable controls
            disableChatControls();
            currentChatId = null;
            sessionStorage.removeItem('currentChatId');
        } finally {
            toggleLoading(false);
            newChatButton.disabled = !dataLoaderSection.style.display === 'none'; // Re-enable only if chat UI is shown
        }
        return newChatId; // Return the ID or null
    }

    // NEW: Fetch server status helper
    async function fetchServerStatus() {
        try {
            const response = await fetch('/status');
            if (!response.ok) {
                console.error('Failed to fetch status:', response.status);
                return null;
            }
            return await response.json();
        } catch (error) {
            console.error("Error fetching server status:", error);
            return null;
        }
    }

    // MODIFIED: Event handler for the New Chat button
    function handleNewChatClick() {
        if (dataLoaderSection.style.display !== 'none') {
            // If data loader is shown, this button shouldn't be active, but handle defensively
            alert("Please upload data first.");
            return;
        }
        createNewChat(); // Call the new function
    }

}); 
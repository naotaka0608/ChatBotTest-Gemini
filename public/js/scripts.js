const API_ENDPOINT = 'http://127.0.0.1:8000/chat';
const USER_ID = 'test-user-001'; 

const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
let currentGeminiMessageElement = null;

function appendMessage(sender, text, isStreaming = false) {
    if (sender === 'Gemini' && isStreaming) {
        currentGeminiMessageElement = document.createElement('div');
        currentGeminiMessageElement.className = 'gemini-message';
        currentGeminiMessageElement.textContent = 'Gemini: ' + text;
        chatBox.appendChild(currentGeminiMessageElement);
    } else if (sender === 'Gemini' && currentGeminiMessageElement) {
        currentGeminiMessageElement.textContent += text;
    } else {
        const msgElement = document.createElement('div');
        msgElement.className = sender === 'User' ? 'user-message' : 'gemini-message';
        msgElement.textContent = `${sender}: ${text}`;
        chatBox.appendChild(msgElement);
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage('User', message);
    userInput.value = '';
    sendButton.disabled = true;

    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: USER_ID, message: message }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`APIエラー: ${response.status} - ${errorText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let firstChunk = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            
            if (firstChunk) {
                appendMessage('Gemini', chunk, true); 
                firstChunk = false;
            } else {
                appendMessage('Gemini', chunk, false);
            }
        }
        
        currentGeminiMessageElement = null;

    } catch (error) {
        console.error('通信エラー:', error);
        appendMessage('Gemini (Error)', '応答中にエラーが発生しました。コンソールを確認してください。');
    } finally {
        sendButton.disabled = false;
        userInput.focus();
    }
}

// Enterキーでの送信を有効にするために、HTMLのインラインイベントを削除した場合はここで設定します
document.getElementById('user-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
// static/script.js
// Function to convert basic markdown to HTML
function simpleMarkdownToHTML(text) {
    // Convert **bold** to <strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Convert `inline code` to <code>
    text = text.replace(/`(.*?)`/g, '<code>$1</code>');

    // Convert numbered lists
    text = text.replace(/^\d+\.\s(.*?)(?=\d+\.|\n|$)/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>)/gm, '<ol>$1</ol>');

    // Convert bullet points (- or *) to unordered lists
    text = text.replace(/^\s*[-*]\s+(.*?)(?=\n|$)/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>)/gm, '<ul>$1</ul>');

    // Convert new lines to <br> (optional, can be removed if paragraph tags are preferred)
    text = text.replace(/\n/g, '<br>');

    return text;
}

// Function to switch between different views
function navigateTo(view) {
    const views = document.querySelectorAll('.view');
    views.forEach(v => v.style.display = 'none'); // Hide all views
    document.getElementById(`view-${view}`).style.display = 'block'; // Show the selected view

    // Update URL without reloading the page
    window.history.pushState({ view }, '', `#${view}`);
}

// When the user presses back or forward in the browser, handle the view
window.onpopstate = function (event) {
    if (event.state && event.state.view) {
        navigateTo(event.state.view);
    } else {
        navigateTo('home');
    }
};

// Load initial view based on the URL
window.onload = function () {
    const currentView = window.location.hash.replace('#', '') || 'home';
    navigateTo(currentView);
};

// Function to handle form submission for file upload and query
document.addEventListener("DOMContentLoaded", () => {
    const responseContainer = document.getElementById("response-container");
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query");
    const fileInput = document.getElementById("file");

    // Function to add a message to the response container
    function addMessage(speaker, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", speaker);

        if (speaker === "user") {
            messageDiv.innerHTML = `<strong>You:</strong> ${simpleMarkdownToHTML(message)}`;
        } else if (speaker === "assistant") {
            messageDiv.innerHTML = `<strong>Assistant:</strong> ${simpleMarkdownToHTML(message)}`;
        }

        responseContainer.appendChild(messageDiv);
        responseContainer.scrollTop = responseContainer.scrollHeight; // Auto-scroll
    }

    // Event listener for form submission
    queryForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const query = queryInput.value.trim();
        if (!query) {
            alert("Please enter a query!");
            return;
        }

        // Display the user's query in the UI
        addMessage("user", query);

        // Check if a file is attached
        if (fileInput.files.length > 0) {
            const formData = new FormData();
            formData.append("query", query);
            formData.append("file", fileInput.files[0]);

            try {
                const response = await fetch("/uploadfile/", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error("Error submitting the form.");
                }

                const result = await response.json();
                addMessage("assistant", result.summary || "No response received.");
            } catch (error) {
                addMessage("assistant", `Error: ${error.message}`);
            }
        } else {
            // No file, send query to /chat/ endpoint
            try {
                const response = await fetch("/chat/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ query }),
                });

                if (!response.ok) {
                    throw new Error("Error submitting the query.");
                }

                const result = await response.json();
                addMessage("assistant", result.response || "No response received.");
            } catch (error) {
                addMessage("assistant", `Error: ${error.message}`);
            }
        }

        // Clear input fields after submission
        queryInput.value = "";
        fileInput.value = "";
    });
});
/*document.getElementById('query-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // Stop the form from refreshing the page

    const query = document.getElementById('query').value.trim();
    const fileInput = document.getElementById('file');
    const responseContainer = document.getElementById('response-container');

    // Ensure query is not empty
    if (!query) {
        console.error("Query is missing");
        responseContainer.innerHTML += `<p>Error: Query is required.</p>`;
        return;
    }

    try {
        if (fileInput.files.length > 0) {
            // File attached, route to /uploadfile/
            const formData = new FormData();
            formData.append('query', query);
            formData.append('file', fileInput.files[0]);

            const response = await fetch('/uploadfile/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error submitting form: ${response.statusText}`);
            }

            const result = await response.json();
            const htmlContent = simpleMarkdownToHTML(result.summary);
            responseContainer.innerHTML += `<div class="formatted-output">${htmlContent}</div>`;
        } else {
            // No file attached, route to /chat/
            const response = await fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error(`Error submitting query: ${response.statusText}`);
            }

            const result = await response.json();
            const htmlContent = simpleMarkdownToHTML(result.response);
            responseContainer.innerHTML += `<div class="formatted-output">${htmlContent}</div>`;
        }

        // Scroll to the bottom of the response container
        responseContainer.scrollTop = responseContainer.scrollHeight;

    } catch (error) {
        console.error("Error in form submission:", error);
        responseContainer.innerHTML += `<p>Error: ${error.message}</p>`;
    }

    // Clear inputs after submission
    fileInput.value = '';
    document.getElementById('query').value = '';
});*/

/*function submitQuery() {
    const query = document.getElementById('query').value;
    const fileInput = document.getElementById('file');
    const responseContainer = document.getElementById('response-container');

    // Proceed with the AJAX request based on the presence of a file or just the query
    if (fileInput.files.length > 0) {
        const formData = new FormData();
        formData.append('query', query);
        formData.append('file', fileInput.files[0]);

        fetch('/uploadfile/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(result => {
            const htmlContent = simpleMarkdownToHTML(result.summary);
            responseContainer.innerHTML += `<div class="formatted-output">${htmlContent}</div><hr>`;
            responseContainer.scrollTop = responseContainer.scrollHeight;
        })
        .catch(error => console.error("Error in form submission:", error));
    } else {
        fetch('/chat/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        })
        .then(response => response.json())
        .then(result => {
            const htmlContent = simpleMarkdownToHTML(result.response);
            responseContainer.innerHTML += `<div class="formatted-output">${htmlContent}</div><hr>`;
            responseContainer.scrollTop = responseContainer.scrollHeight;
        })
        .catch(error => console.error("Error in form submission:", error));
    }

    // Clear inputs
    fileInput.value = '';
    document.getElementById('query').value = '';
}*/

const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file');

// Drag and drop functionality
dropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dropArea.classList.remove('dragover');
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;  // This attaches the files to the file input
    }
});

// Manual file selection using "Choose a file" button
dropArea.addEventListener('click', () => {
    fileInput.click();  // Trigger file input click
});

// Prevent the default click action from re-triggering
fileInput.addEventListener('click', (event) => {
    event.stopPropagation();  // Prevent the file input from reopening the file dialog
});
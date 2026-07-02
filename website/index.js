document.addEventListener("DOMContentLoaded", () => {
    const navItems = document.querySelectorAll(".nav-item");
    const contentArea = document.getElementById("content");

    // Load document content dynamically
    async function loadDoc(docName) {
        contentArea.innerHTML = `<p style="color: var(--text-secondary);">Loading ${docName} guide...</p>`;
        
        try {
            // Load from docs folder relative to current location
            const response = await fetch(`docs/${docName}.md`);
            if (!response.ok) {
                throw new Error(`Failed to load: docs/${docName}.md`);
            }
            const markdownText = await response.text();
            
            // Parse Markdown to HTML
            const parsedHtml = marked.parse(markdownText);
            contentArea.innerHTML = parsedHtml;

            // Apply Prism syntax highlighting
            Prism.highlightAllUnder(contentArea);

            // Add Copy Buttons to code block elements
            addCopyButtons();

        } catch (error) {
            contentArea.innerHTML = `
                <h1>Error Loading Document</h1>
                <p>Could not retrieve the documentation file. Make sure you are serving the files via a local HTTP server.</p>
                <blockquote style="border-color: #ef4444; background: rgba(239, 68, 68, 0.05);">
                    <p style="color: #ef4444;">${error.message}</p>
                </blockquote>
            `;
        }
    }

    // Add Copy Action Button to all <pre> elements
    function addCopyButtons() {
        const preBlocks = document.querySelectorAll("pre");
        preBlocks.forEach((pre) => {
            const button = document.createElement("button");
            button.className = "copy-btn";
            button.textContent = "Copy";
            
            pre.appendChild(button);

            button.addEventListener("click", () => {
                const codeText = pre.querySelector("code").textContent;
                navigator.clipboard.writeText(codeText).then(() => {
                    button.textContent = "Copied!";
                    button.classList.add("copied");
                    setTimeout(() => {
                        button.textContent = "Copy";
                        button.classList.remove("copied");
                    }, 2000);
                });
            });
        });
    }

    // Add navigation click event handlers
    navItems.forEach((item) => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            
            // Toggle active styling
            navItems.forEach((nav) => nav.classList.remove("active"));
            item.classList.add("active");

            // Fetch and render document content
            const docName = item.getAttribute("data-doc");
            loadDoc(docName);
        });
    });

    // Default home document load
    loadDoc("environments");
});

async function addRepo() {
    const repoName = window.REPO_NAME;

    try {
        const response = await fetch('/api/add_repo/' + encodeURIComponent(repoName), {
            method: 'POST',
        });

        if (response.ok) {
            const loadingContainer = document.getElementById('loading-container');
            const root = document.getElementById('root');
            loadingContainer.classList.add('hidden');
            root.classList.add('show');
        } else {
            const errorData = await response.json();
            const errorDiv = document.getElementById('error');
            const loadingDiv = document.getElementById('loading');
            errorDiv.textContent = errorData.detail || 'An error occurred while adding the repository';
            errorDiv.classList.add('show');
            loadingDiv.style.display = 'none';
        }
    } catch (error) {
        const errorDiv = document.getElementById('error');
        const loadingDiv = document.getElementById('loading');
        errorDiv.textContent = 'Error: ' + error.message;
        errorDiv.classList.add('show');
        loadingDiv.style.display = 'none';
    }
}

addRepo();

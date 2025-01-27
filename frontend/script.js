window.addEventListener('load', () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear-button');
    const predictButton = document.getElementById('predict-button');
    const csvInput = document.getElementById('csv-input');
    const uploadCsvButton = document.getElementById('upload-csv-button');
    const predictionResult = document.getElementById('prediction-result');
    let isDrawing = false;
    let strokes = [];

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        strokes.push([]);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDrawing) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            strokes[strokes.length - 1].push([x, y]);
            ctx.lineTo(x, y);
            ctx.stroke();
        }
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
        ctx.beginPath();
    });

    clearButton.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        strokes = [];
    });

    predictButton.addEventListener('click', async () => {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ strokes })
        });

        const result = await response.json();
        predictionResult.innerText = `Prediction: ${result.prediction}`;
    });

    uploadCsvButton.addEventListener('click', async () => {
        const file = csvInput.files[0];
        if (!file) {
            alert("Please select a CSV file.");
            return;
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            const csvData = e.target.result;
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ csv_data: csvData })
            });

            const result = await response.json();
            predictionResult.innerText = `Prediction: ${result.prediction}`;
        };
        reader.readAsText(file);
    });
});

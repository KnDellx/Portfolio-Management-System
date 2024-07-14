document.getElementById('forecast-report-btn').addEventListener('click', function () {
    $('#forecastReportModal').modal('show');
});

document.getElementById('forecast-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    const stockCode = document.getElementById('stock-code').value;
    const startDate = document.getElementById('start-date').value;
    const predictionDays = document.getElementById('prediction-days').value;
    
    // Call your backend API to get forecast data
    fetch('/api/getForecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ stockCode, startDate, predictionDays })
    })
    .then(response => response.json())
    .then(data => {
        // Assuming data contains { lstm, randomForest, lstmMSE, randomForestMSE }
        document.getElementById('forecast-results').style.display = 'block';

        // Render LSTM chart
        new Chart(document.getElementById('lstm-chart'), {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'LSTM Forecast',
                    data: data.lstm,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }]
            },
            options: { responsive: true }
        });

        // Render Random Forest chart
        new Chart(document.getElementById('random-forest-chart'), {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Random Forest Forecast',
                    data: data.randomForest,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2
                }]
            },
            options: { responsive: true }
        });

        // Display MSE values
        document.getElementById('lstm-mse').textContent = 'LSTM MSE: ' + data.lstmMSE;
        document.getElementById('random-forest-mse').textContent = 'Random Forest MSE: ' + data.randomForestMSE;
    });
});

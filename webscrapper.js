// Function to pause execution for a given amount of milliseconds
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Async function to download data
async function downloadData() {
    // Get all pins (markers) on the map
    let pins = document.querySelector('.leaflet-marker-pane').childNodes;

    for (let i = 0; i < pins.length; i++) {
        // Click on each pin to load the associated data
        pins[i].click();

        // Wait for 1 second to allow the dynamic content to load
        await sleep(1000);

        // Extract location information from the dialog header
        let h2Content = document.querySelector('#dialogheader').textContent;
        const number = h2Content.match(/Lokalitet:\s(\d+)/)[1];
        const lokalitetId = parseInt(number);

        // Get the list of locations from the 'vm' object
        let lokaliteter = vm.lokaliteter();

        // Find the corresponding latitude and longitude using the location ID
        const obj = lokaliteter.find(o => o.LokalitetID === lokalitetId);
        const latitude = obj.Latitude;
        const longitude = obj.Longitude;

        // Add latitude and longitude to each object in the 'vm.media' array
        let json = vm.media.map(obj => ({
            ...obj,
            latitude: latitude,
            longitude: longitude
        }));

        // Create a blob from the JSON object
        let blob = new Blob([JSON.stringify(json)], { type: 'application/json' });

        // Create a download link for the JSON file
        let downloadLink = document.createElement('a');
        downloadLink.href = URL.createObjectURL(blob);
        downloadLink.download = 'data' + i + '.json';
        downloadLink.click(); // Trigger the download
    }
}

// Start the data download process
downloadData();
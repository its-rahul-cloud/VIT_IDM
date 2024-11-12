<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Deep Learning Project - Virtual Try-On System</title>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  <script>
    const apiUrl = <Ngrok-Link>;
    const skipBrowserWarningHeader = {
      "ngrok-skip-browser-warning": "69420"
    };

    // Check server status by sending a request to '/hello'
    async function checkServerStatus() {
      try {
        const response = await fetch(`${apiUrl}hello`, {
          method: "GET",
          headers: new Headers(skipBrowserWarningHeader), // Include ngrok header
        });
        const data = await response.text();

        if (data === "yes") {
          fetchImages();  // Proceed to fetch images if server is live
        } else {
          document.getElementById("content").innerHTML = "Server is not live. Please try again later.";
        }
      } catch (error) {
        document.getElementById("content").innerHTML = "Server not live. Please check the server connection.";
      }
    }

    // Call the checkServerStatus function on page load
    window.onload = checkServerStatus;

    // Fetch available images and populate dropdowns
    async function fetchImages() {
      const response = await fetch(`${apiUrl}/list_images`, {
        method: "GET",
        headers: new Headers(skipBrowserWarningHeader), // Include ngrok header
      });
      const data = await response.json();

      const bodySelect = document.getElementById("bodySelect");
      const clothSelect = document.getElementById("clothSelect");

      bodySelect.innerHTML = "<option value=''>Select Body Image</option>";
      clothSelect.innerHTML = "<option value=''>Select Cloth Image</option>";

      data.body_images.forEach(img => {
        const option = document.createElement("option");
        option.value = img;
        option.text = img;
        bodySelect.appendChild(option);
      });

      data.cloth_images.forEach(img => {
        const option = document.createElement("option");
        option.value = img;
        option.text = img;
        clothSelect.appendChild(option);
      });

      // Show the content section after fetching images
      document.getElementById("content").classList.remove("hidden");
    }

    // Preview selected image
    async function previewImage(type) {
      const selectElement = document.getElementById(type + "Select");
      const selectedImage = selectElement.value;

      if (selectedImage) {
        const imageUrl = `${apiUrl}${type === "body" ? "image" : "cloth"}/${selectedImage}`;

        // Create a new image element dynamically
        const img = new Image();
        
        // Set the CORS policy
        img.crossOrigin = 'anonymous'; // Allows CORS if server supports it

        try {
          const response = await fetch(imageUrl, {
            method: 'GET',
            headers: new Headers({
              'ngrok-skip-browser-warning': '69420', // Bypass ngrok warning
            }),
          });

          if (response.ok) {
            const imageBlob = await response.blob(); // Get the image as a Blob
            const imageObjectURL = URL.createObjectURL(imageBlob); // Create a URL for the image blob

            img.src = imageObjectURL;

            img.onload = () => {
              const previewElement = document.getElementById(type + "Preview");
              previewElement.src = img.src;
              previewElement.classList.remove("hidden"); // Show the image
            };
          } else {
            throw new Error('Image request failed');
          }
        } catch (error) {
          console.error('Failed to load image:', error);
          const previewElement = document.getElementById(type + "Preview");
          previewElement.src = '/path/to/default-image.jpg'; // Fallback in case of an error
          previewElement.classList.remove("hidden"); // Show the fallback image
        }
      }
    }

    // Upload custom image (body or cloth)
    async function uploadImage(imageType) {
      const fileInput = document.getElementById(`${imageType}Upload`);
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      const response = await fetch(`${apiUrl}/upload/${imageType}`, {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      alert(result.message);
      fetchImages(); // Re-fetch images to update dropdown
    }

    async function fetchImageWithHeader(imageUrl) {
  try {
    const response = await fetch(imageUrl, {
      method: 'GET',
      headers: new Headers(skipBrowserWarningHeader),  // Include Ngrok header to skip the browser warning
    });

    if (!response.ok) {
      throw new Error('Failed to load image');
    }

    const imageBlob = await response.blob();
    return URL.createObjectURL(imageBlob); // Return a URL for the image Blob
  } catch (error) {
    console.error('Error fetching image:', error);
    return '/path/to/default-image.jpg'; // Fallback image
  }
}

async function mergeImages() {
  const bodyImage = document.getElementById("bodySelect").value;
  const clothImage = document.getElementById("clothSelect").value;

  if (!bodyImage || !clothImage) {
    alert("Please select both body and cloth images.");
    return;
  }

  const response = await fetch(`${apiUrl}merge_images`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "69420",  // Include Ngrok header to skip the browser warning
    },
    body: JSON.stringify({
      body_image: bodyImage,
      cloth_image: clothImage,
    }),
  });

  const result = await response.json();
  alert(result.message);

  // Fetch original body and cloth images with the custom header
  const bodyImageUrl = `${apiUrl}results/first_stage/${result.result_image}`;
  const clothImageUrl = `${apiUrl}results/second_stage/${result.result_image}`;
//   const resultImageUrl = `${apiUrl}${result.result_image}`;

  // Fetch images with skip-browser-warning header
  const resultBodyImgUrl = await fetchImageWithHeader(bodyImageUrl);
  const resultClothImgUrl = await fetchImageWithHeader(clothImageUrl);
//   const resultImgUrl = await fetchImageWithHeader(resultImageUrl);

  // Show original body image
  const resultBodyImg = document.getElementById("resultBodyImage");
  resultBodyImg.src = resultBodyImgUrl;
  resultBodyImg.classList.remove("hidden");

  // Show original cloth image
  const resultClothImg = document.getElementById("resultClothImage");
  resultClothImg.src = resultClothImgUrl;
  resultClothImg.classList.remove("hidden");

//   // Show merged result
//   const resultImg = document.getElementById("resultImage");
//   resultImg.src = resultImgUrl;
//   resultImg.classList.remove("hidden");
}

  </script>
</head>
<body class="bg-gray-100 text-gray-900 p-6">
  <div id="content" class="container mx-auto hidden">
    <h1 class="text-3xl font-bold mb-2 text-center">Deep Learning Project</h1>
    <h2 class="text-xl font-semibold mb-4 text-center">Virtual Try-On System</h2>
    <p class="text-gray-700 mb-8 text-center">Implemented and improved by Archana, Bitanuka, Nirban, Rahul, Ram Vikas</p>

    <div class="flex flex-col md:flex-row gap-8">
      <!-- Left column for Body Image -->
      <div class="w-full md:w-1/2">
        <label for="bodySelect" class="block text-gray-700 mb-2">Select Body Image:</label>
        <select id="bodySelect" class="w-full p-2 border rounded-md" onchange="previewImage('body')"></select>
        <div class="mt-4">
          <label class="block text-gray-700">Or upload a custom body image:</label>
          <input type="file" id="bodyUpload" class="w-full p-2 border rounded-md mb-2">
          <button onclick="uploadImage('body')" class="bg-blue-500 text-white px-4 py-2 rounded-md">Upload Body Image</button>
        </div>
        <div class="mt-4">
      <img id="bodyPreview" class="hidden w-1/2 mx-auto rounded-md shadow-md" alt="Selected Body Image Preview">
    </div>
      </div>

      <!-- Right column for Cloth Image -->
      <div class="w-full md:w-1/2">
        <label for="clothSelect" class="block text-gray-700 mb-2">Select Cloth Image:</label>
        <select id="clothSelect" class="w-full p-2 border rounded-md" onchange="previewImage('cloth')"></select>
        <div class="mt-4">
          <label class="block text-gray-700">Or upload a custom cloth image:</label>
          <input type="file" id="clothUpload" class="w-full p-2 border rounded-md mb-2">
          <button onclick="uploadImage('cloth')" class="bg-blue-500 text-white px-4 py-2 rounded-md">Upload Cloth Image</button>
        </div>
        <div class="mt-4">
      <img id="clothPreview" class="hidden w-1/2 mx-auto rounded-md shadow-md" alt="Selected Cloth Image Preview">
    </div>
      </div>
    </div>

    <!-- Merge Button -->
    <div class="mt-8 text-center">
      <button onclick="mergeImages()" class="bg-green-500 text-white px-4 py-2 rounded-md">Merge Body and Cloth</button>
    </div>

    <!-- Display Result Image -->
    <div class="mt-8">
      <h3 class="text-lg font-semibold mb-2 text-center">Original and Merged Results:</h3>
      
  <div class="text-center flex justify-center space-x-4">
  <div>
    <h4>IDM_VTON Image:</h4>
    <img id="resultBodyImage" class="hidden w-full max-w-md mx-auto rounded-lg shadow-md" alt="Original Body Image">
  </div>
  
  <div>
    <h4>Fine-Tuned IDM Image:</h4>
    <img id="resultClothImage" class="hidden w-full max-w-md mx-auto rounded-lg shadow-md" alt="Original Cloth Image">
  </div>
</div>

    <!-- Show the merged result -->
   <img id="resultImage" class="hidden w-full max-w-md mx-auto rounded-lg shadow-md" alt="Merged Result">
    </div>
  </div>

  <div id="serverError" class="text-center mt-8 text-red-500 font-bold hidden">
    Server not live. Please try again later.
  </div>

  <script>
    checkServerStatus();
  </script>
</body>
</html>

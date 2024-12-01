# AI+BII: Building Image Inpainting and Style Transfer

**AI+BII** is an innovative project that leverages generative AI (GenAI) to restore images of historical buildings. It allows users to explore how damaged structures might have looked in their original, intact form. Additionally, the platform provides customizable style transfer options, enabling users to experiment with various artistic effects.

---

## Features

1. **Building Restoration**  
   - Restore damaged building images using AI-powered inpainting.
   - Leverages the `stabilityai/stable-diffusion-2-inpainting` model for precise and realistic results.

2. **Style Transfer**  
   - Apply artistic styles to the restored images. Current supported styles include:
     - Watercolor
     - Architectural Sketch
     - Pixel Art
     - Cyberpunk

3. **Dynamic Parameter Adjustment**  
   - Users can adjust parameters like color saturation, contrast, and pixel size to customize results.

---

## File Overview

- **`app.py`**  
  A Flask-based backend implementation that handles image restoration and style transfer requests. This script is designed to run on Google Colab.

- **`index-v10.html`**  
  The frontend interface where users can upload images, input descriptions, and view results.

---

## Prerequisites

1. **Python Environment**  
   Ensure you have Python 3.8 or above installed locally or access to Google Colab.

2. **Ngrok Installation**  
   Download and install [Ngrok](https://ngrok.com/), and set up your authentication token.

3. **Frontend Browser Support**  
   A modern browser to interact with the frontend HTML file.

---

## Setup Instructions

### Backend

1. **Edit Authentication Token**  
   Open `app.py` and replace the `ngrok.set_auth_token` line with your personal Ngrok token:
   ```python
   ngrok.set_auth_token("your_ngrok_token_here")
   ```

2. **Run the Backend**  
   Execute `app.py` in your local Python environment or on Google Colab. Once the server starts, Ngrok will generate a public URL. Note down this URL:
   ```
   Public URL: NgrokTunnel: "https://your-ngrok-url.ngrok-free.app" -> "http://localhost:5000"
   ```

### Frontend

1. **Update API Endpoints**  
   Modify the two API endpoints in `index-v10.html` to match the Ngrok URL from the backend:

   - For the `style_transfer` endpoint:
     ```javascript
     const response = await fetch("https://your-ngrok-url.ngrok-free.app/style_transfer", {
         method: "POST",
         headers: {
             "Content-Type": "application/json",
         },
         body: JSON.stringify(payload),
     });
     ```

   - For the `generate` endpoint:
     ```javascript
     const response = await fetch("https://your-ngrok-url.ngrok-free.app/generate", { 
         method: "POST", 
         body: formData 
     });
     ```

2. **Open Frontend**  
   Launch `index-v10.html` in your browser to interact with the platform.

---

## Usage

1. **Upload an Image**  
   Choose a damaged building image and upload it through the interface.

2. **Input a Description**  
   Provide a detailed description of the building's original appearance or desired style, e.g.,  
   ```
   A restored Gothic cathedral with vibrant stained-glass windows.
   ```

3. **Generate Results**  
   Click the "Generate" button to view AI-restored images.

4. **Apply Style Transfer**  
   Select a restored image and apply a style from the available options using the "Apply Style Transfer" button.

---

## Notes

- The backend is designed to run on Google Colab for ease of access to GPU resources.
- Each time the backend restarts, a new Ngrok URL is generated. Ensure the frontend endpoints are updated accordingly.
- If you encounter memory issues on Colab, consider reducing the image size or the number of inference steps.

---

## Future Improvements

1. **Expand Style Options**  
   Add more customizable styles to cater to diverse user preferences.

2. **Enhanced Model Training**  
   Improve restoration accuracy for severely damaged structures.

3. **Cloud Deployment**  
   Host the platform on a cloud server to support a larger audience.

---

We hope you enjoy using **AI+BII**! For questions or feedback, feel free to reach out to the development team.

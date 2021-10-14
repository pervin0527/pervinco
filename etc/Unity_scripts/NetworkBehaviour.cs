using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

public class NetworkBehaviour : MonoBehaviour
{
    // Public fields
    public float ProbabilityThreshold = 0.5f;
    public Vector2 InputFeatureSize = new Vector2(320, 320); // model의 input_shape에 따라 변경
    public Text StatusBlock;

    // Private fields
    private NetworkModel _networkModel;
    private MediaCaptureUtility _mediaCaptureUtility;
    private bool _isRunning = false;

    #region UnityMethods
    async void Start()
    {
        try
        {
            // Create a new instance of the network model class
            // and asynchronously load the onnx model
            _networkModel = new NetworkModel(); // model load
            await _networkModel.LoadModelAsync();
            StatusBlock.text = $"Loaded model. Starting camera...";

#if ENABLE_WINMD_SUPPORT
            // Configure camera to return frames fitting the model input size
            try
            {
                Debug.Log("Creating MediaCaptureUtility and initializing frame reader.");
                _mediaCaptureUtility = new MediaCaptureUtility();
                await _mediaCaptureUtility.InitializeMediaFrameReaderAsync(
                    (uint)InputFeatureSize.x, (uint)InputFeatureSize.y);
                StatusBlock.text = $"Camera started. Running!";

                Debug.Log("Successfully initialized frame reader.");
            }
            catch (Exception ex)
            {
                StatusBlock.text = $"Failed to start camera: {ex.Message}. Using loaded/picked image.";

            }

            // Run processing loop in separate parallel Task, get the latest frame
            // and asynchronously evaluate
            Debug.Log("Begin performing inference in frame grab loop.");

            _isRunning = true;
            await Task.Run(async () =>
            {
                while (_isRunning)
                {
                    if (_mediaCaptureUtility.IsCapturing)
                    {
                        using (var videoFrame = _mediaCaptureUtility.GetLatestFrame())
                        {
                            await EvaluateFrame(videoFrame);
                        }
                    }
                    else
                    {
                        return;
                    }
                }
            });
#endif 
        }
        catch (Exception ex)
        {
            StatusBlock.text = $"Error init: {ex.Message}";
            Debug.LogError($"Failed to start model inference: {ex}");
        }
    }

    private async void OnDestroy()
    {
        _isRunning = false;
        if (_mediaCaptureUtility != null)
        {
            await _mediaCaptureUtility.StopMediaFrameReaderAsync();
        }
    }
    #endregion

#if ENABLE_WINMD_SUPPORT
    private async Task EvaluateFrame(Windows.Media.VideoFrame videoFrame)
    {
        try
        {
            // Get the current network prediction from model and input frame
            var result = await _networkModel.EvaluateVideoFrameAsync(videoFrame);

            // Update the UI with prediction
            UnityEngine.WSA.Application.InvokeOnAppThread(() =>
            {
                //StatusBlock.text = $"Label: {result.PredictionLabel} " +
                //$"Probability: {Math.Round(result.PredictionProbability, 3) * 100}% " +
                //$"Inference time: {result.PredictionTime} ms";
                
                // StatusBlock.text = $"TESTING";
                // 결과값 출력
                StatusBlock.text = $"bboxes : {result.PredictionBoxes}" +
                                   $"classes : {result.PredictionClasses}" +
                                   $"scores : {result.PredictionScores}";

            }, false);
        }
        catch (Exception ex)
        {
            Debug.Log($"Exception {ex}");
        }
    }

#endif
}
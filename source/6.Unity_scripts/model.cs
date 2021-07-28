﻿#if ENABLE_WINMD_SUPPORT
using System;
using System.Threading.Tasks;
using Windows.Storage.Streams;
using Microsoft.AI.MachineLearning;

public sealed class CustomNetworkInput
{
    public ImageFeatureValue features;
}

public sealed class CustomNetworkOutput
{
    // public TensorFloat prediction;
    public TensorFloat bboxes;
    public TensorFloat classes;
    public TensorFloat scores;
}

public sealed class CustomNetworkModel
{
    private LearningModel model;
    private LearningModelSession session;
    private LearningModelBinding binding;
    public static async Task<CustomNetworkModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
    {
        // Run on the GPU
        //var device = new LearningModelDevice(LearningModelDeviceKind.DirectX);
         
        CustomNetworkModel learningModel = new CustomNetworkModel();
        learningModel.model = await LearningModel.LoadFromStreamAsync(stream);
        //learningModel.session = new LearningModelSession(learningModel.model, device);
        learningModel.session = new LearningModelSession(learningModel.model);
        learningModel.binding = new LearningModelBinding(learningModel.session);
        return learningModel;
    }

    public async Task<CustomNetworkOutput> EvaluateAsync(CustomNetworkInput input)
    {
        // Ensure the input and output fields are bound to the correct
        // layer names in the onnx model
        binding.Bind("input_tensor:0", input.features);
        var result = await session.EvaluateAsync(binding, "0");
        var output = new CustomNetworkOutput();
        
        // output.prediction = result.Outputs["detection_classes"] as TensorFloat;
        output.bboxes = result.Outputs["detection_boxes"] as TensorFloat;
        output.classes = result.Outputs["detection_classes"] as TensorFloat;
        output.scores = result.Outputs["detection_scores"] as TensorFloat;

        return output;
    }
}

#endif

#FAQ
Here we discuss frequent questions that may occur and we found useful along the way.

1- Does FSDP support mixed precision in one FSDP unit? meaning in one FSDP unit some of your parameters are in Fp16/Bf16 and others in FP32.

FSDP requires each FSDP unit to have consistent precision, so this case is not support at this point. It might be added in future but no ETA.

2-  How FSDP handles mixed grad requirements? FSDP does not support of mixed `require_grad` in one FSDP unit. This means if you are planning to freeze some layers, need to do it on FSDP unit level rather model layer. In this particular case, let assume our model has 30 decoder layers and we want to freeze the bottom 28 layers and only train 2 top transformer layers. In this sense, we need to make sure `require_grad` for the top two transformer layers are set to `True`.

3- How PEFT methods work with FSDP in terms of grad requirements/ layer freezing? We wrap the PEFT modules separate from transfromer layer in auto_wrapping policy, that would result in PEFT models having `require_grad=True` while the rest of the model is  `require_grad=False`.
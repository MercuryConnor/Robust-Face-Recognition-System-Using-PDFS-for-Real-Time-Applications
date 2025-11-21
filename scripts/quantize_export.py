"""Export and (simple) quantization utilities.

Exports a TorchScript model (FP32 or FP16) and optionally an ONNX file.
If ONNX quantization is available, attempts dynamic quantization.
"""
import argparse
import os
import torch
from models.inference import load_checkpoint


def export_torchscript(backbone, classifier, device, out_path, fp16=False):
    backbone = backbone.to(device).eval()
    example = torch.randn(1,3,112,112).to(device)
    if fp16:
        backbone = backbone.half()
        example = example.half()
    traced = torch.jit.trace(backbone, example)
    traced.save(out_path)


def export_onnx(backbone, out_path, opset=12):
    backbone = backbone.eval()
    import torch
    example = torch.randn(1,3,112,112)
    torch.onnx.export(backbone, example, out_path, export_params=True, opset_version=opset,
                      input_names=['input'], output_names=['output'])


def try_onnx_quantize(onnx_path, quantized_out):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, quantized_out, weight_type=QuantType.QInt8)
        return True
    except Exception as ex:
        print('ONNX dynamic quantization not available:', ex)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--onnx', help='also export ONNX to this path')
    args = p.parse_args()

    backbone, classifier, class_to_idx, device = load_checkpoint(args.model)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export_torchscript(backbone, classifier, device, args.out, fp16=args.fp16)
    print('Saved TorchScript to', args.out)
    if args.onnx:
        export_onnx(backbone, args.onnx)
        print('Saved ONNX to', args.onnx)
        qout = os.path.splitext(args.onnx)[0] + '.quant.onnx'
        if try_onnx_quantize(args.onnx, qout):
            print('Saved quantized ONNX to', qout)


if __name__ == '__main__':
    main()
# Quantize and export script

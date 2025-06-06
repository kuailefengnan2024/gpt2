import torch
import torch.nn.functional as F

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name()}")

# 检查scaled_dot_product_attention可用的后端
print("\n=== Flash Attention后端检查 ===")
try:
    # 创建测试tensor
    device = 'cuda'
    q = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
    
    # 检查可用的后端
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
        try:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print("✅ Flash Attention可用")
        except:
            print("❌ Flash Attention不可用")
    
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
        try:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print("✅ Memory-efficient Attention可用")
        except:
            print("❌ Memory-efficient Attention不可用")
            
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        try:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print("✅ Math Attention可用")
        except:
            print("❌ Math Attention不可用")
            
except Exception as e:
    print(f"检查时出错: {e}")
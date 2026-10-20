// ImpactTester.cs
// 破坏测试：鼠标点击场景物体 → 在点击点施加能量 → 观察「撞不同位置坏不同部位」。
// 需配合 GameSceneLoader 生成的可破坏物体（DestructibleObject）。
// 挂到 Main Camera 或任意空物体上，运行时鼠标左键点击即可测试。
using UnityEngine;

public class ImpactTester : MonoBehaviour
{
    [Tooltip("每次点击施加的撞击能量(J)，默认 270J 对应论文 4.4 表")]
    public float energy = 270f;

    [Tooltip("撞击作用半径(米)，影响该半径内的部位")]
    public float radius = 0.5f;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit, 200f))
            {
                DestructibleObject d = hit.collider.GetComponentInParent<DestructibleObject>();
                if (d == null)
                {
                    Debug.Log($"[ImpactTester] 命中 {hit.collider.name}，但无 DestructibleObject");
                    return;
                }
                Debug.Log($"[ImpactTester] 撞击 {hit.collider.name} @ {hit.point} 能量 {energy}J");
                d.ApplyImpact(hit.point, energy, radius);
                Debug.Log($"[ImpactTester] → 状态={d.state}, 破碎比例={d.brokenFraction:F2}");
            }
        }
    }
}

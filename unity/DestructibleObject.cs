// DestructibleObject.cs
// 位置相关破坏组件：复刻 utils/damage.py 的逻辑。
//   - 物体被分成若干子部位，每部位有 strength/hp。
//   - 撞击只作用于撞击点附近部位 → 撞不同位置坏不同部位。
//   - damage_response: none(不可破坏) / deform(只变形不碎) / break(局部断) / shatter(整体碎)。
// 破坏视觉回调（可派生或挂事件）：OnPartBroken(part) 用于生成碎块、换材质、播放特效等。
using System;
using System.Collections.Generic;
using UnityEngine;

public class DestructibleObject : MonoBehaviour
{
    [Serializable]
    public class Part
    {
        public Vector3 center;
        public float strength;
        public float hp;
        public float damage;
        public bool broken;
    }

    public string damageResponse = "deform";
    public string state = "intact";
    public float brokenFraction = 0f;
    public List<Part> parts = new List<Part>();
    public float destroyFraction = 0.5f;

    public event Action<Part> OnPartBroken;

    public void Init(DamageDef def)
    {
        damageResponse = def.damage_response;
        state = def.state;
        parts.Clear();
        foreach (PartDef p in def.parts)
        {
            parts.Add(new Part
            {
                center = new Vector3(p.center[1], p.center[2], p.center[0]),  // 同坐标转换
                strength = p.strength,
                hp = p.hp,
                damage = p.damage,
                broken = p.state == "broken",
            });
        }
        UpdateState();
    }

    public void ApplyImpact(Vector3 point, float energy, float radius = 0.5f)
    {
        if (damageResponse == "none") return;   // 刚性不可破坏

        // 找受影响部位（半径内；否则最近）
        List<int> affected = new List<int>();
        float best = float.MaxValue;
        int nearest = -1;
        for (int i = 0; i < parts.Count; i++)
        {
            float d = Vector3.Distance(parts[i].center, point);
            if (d < best) { best = d; nearest = i; }
            if (d <= radius) affected.Add(i);
        }
        if (affected.Count == 0 && nearest >= 0) affected.Add(nearest);

        bool anyBroken = false;
        foreach (int i in affected)
        {
            Part p = parts[i];
            if (p.broken) continue;
            p.damage += energy / Mathf.Max(p.hp, 1e-6f);

            if (damageResponse == "shatter" && p.damage >= 1f)
            {
                foreach (Part q in parts) { q.broken = true; }
                anyBroken = true;
                break;
            }
            if (p.damage >= 1f)
            {
                p.broken = true;
                anyBroken = true;
                OnPartBroken?.Invoke(p);
            }
        }
        UpdateState();
    }

    void UpdateState()
    {
        int broken = 0;
        foreach (Part p in parts) if (p.broken) broken++;
        brokenFraction = parts.Count > 0 ? (float)broken / parts.Count : 0f;
        state = brokenFraction >= destroyFraction ? "destroyed" :
                brokenFraction > 0 ? "damaged" : "intact";
    }
}

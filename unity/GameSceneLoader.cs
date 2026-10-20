// GameSceneLoader.cs
// 读取 scripts/export_game_scene.py 生成的 game_scene.json，在 Unity 中重建场景：
//   每个物体 = GameObject + BoxCollider + Rigidbody(PhysicMaterial) + DestructibleObject
// 坐标约定：nuScenes (x前, y左, z上) → Unity (x右, y上, z前)，
//   即 Unity = (nuScenes.y, nuScenes.z, nuScenes.x)。
//
// 用法：把本文件 + DestructibleObject.cs 放进 Assets/，在任意 GameObject 上挂 GameSceneLoader，
//   把 game_scene.json 拖进 jsonAsset（或放入 StreamingAssets 并填 jsonPath），Play 即自动生成。
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable]
public class GameSceneRoot
{
    public int num_objects;
    public float impact_energy_used;
    public List<GameObjectDef> objects;
}

[Serializable]
public class GameObjectDef
{
    public int id;
    public int class_id;
    public string class_name;
    public string physical;
    public string safety;
    public float lethality;
    public int n_points;
    public ColliderDef collider;
    public RigidbodyDef rigidbody;
    public DamageDef damage;
}

[Serializable]
public class ColliderDef
{
    public string type;
    public float[] center;
    public float[] bbox_min;
    public float[] bbox_max;
}

[Serializable]
public class RigidbodyDef
{
    public bool is_static;
    public float mass;
    public float friction;
    public float restitution;
}

[Serializable]
public class DamageDef
{
    public int class_id;
    public string physical;
    public string damage_response;
    public bool is_static;
    public float mass;
    public string state;
    public float broken_fraction;
    public List<PartDef> parts;
}

[Serializable]
public class PartDef
{
    public int idx;
    public float[] center;
    public float strength;
    public float hp;
    public float damage;
    public string state;
    public float geometric_factor;
}

public class GameSceneLoader : MonoBehaviour
{
    [Tooltip("game_scene.json 的路径（相对 StreamingAssets，如 \"game_scene.json\"）")]
    public string jsonPath = "game_scene.json";

    [Tooltip("生成物体的父节点（空则挂到本物体下）")]
    public Transform parent;

    void Start()
    {
        LoadScene();
    }

    public void LoadScene()
    {
        string json = null;
        if (!string.IsNullOrEmpty(jsonPath))
        {
            string full = Path.Combine(Application.streamingAssetsPath, jsonPath);
            if (File.Exists(full)) json = File.ReadAllText(full);
        }
        if (string.IsNullOrEmpty(json))
        {
            Debug.LogError("未找到 game_scene.json，请放到 StreamingAssets 或重设 jsonPath");
            return;
        }

        GameSceneRoot root = JsonUtility.FromJson<GameSceneRoot>(json);
        Transform rootT = parent != null ? parent : transform;
        foreach (GameObjectDef def in root.objects)
        {
            SpawnObject(def, rootT);
        }
        Debug.Log($"[GameSceneLoader] 生成 {root.objects.Count} 个物体");
    }

    static Vector3 ToUnity(float[] v) => v == null ? Vector3.zero : new Vector3(v[1], v[2], v[0]);

    void SpawnObject(GameObjectDef def, Transform rootT)
    {
        GameObject go = new GameObject($"{def.class_name}_{def.id}");
        go.transform.SetParent(rootT, false);

        // BoxCollider：由 bbox 得 center + size
        Vector3 bmin = ToUnity(def.collider.bbox_min);
        Vector3 bmax = ToUnity(def.collider.bbox_max);
        BoxCollider bc = go.AddComponent<BoxCollider>();
        bc.center = (bmin + bmax) * 0.5f;
        bc.size = new Vector3(
            Mathf.Abs(bmax.x - bmin.x),
            Mathf.Abs(bmax.y - bmin.y),
            Mathf.Abs(bmax.z - bmin.z));

        // Rigidbody + PhysicMaterial
        Rigidbody rb = go.AddComponent<Rigidbody>();
        rb.mass = def.rigidbody.mass;
        rb.isKinematic = def.rigidbody.is_static;   // 建筑/路面等静态 → kinematic
        rb.useGravity = !def.rigidbody.is_static;
        if (def.rigidbody.friction > 0 || def.rigidbody.restitution > 0)
        {
            PhysicMaterial pm = new PhysicMaterial();
            pm.dynamicFriction = def.rigidbody.friction;
            pm.staticFriction = def.rigidbody.friction;
            pm.bounciness = def.rigidbody.restitution;
            pm.frictionCombine = PhysicMaterialCombine.Average;
            bc.material = pm;
        }

        // 破坏组件（位置相关损坏逻辑）
        DestructibleObject d = go.AddComponent<DestructibleObject>();
        d.Init(def.damage);
    }
}

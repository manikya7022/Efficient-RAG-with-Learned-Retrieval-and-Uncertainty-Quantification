# Troubleshooting

## Common Issues

### Ollama Connection Failed

**Symptoms:** `Connection refused` when calling LLM

**Solutions:**
```bash
# Check logs
docker logs ollama_phd

# Restart service
docker-compose restart ollama

# Re-pull model
docker exec ollama_phd ollama pull llama3.2:3b
```

### ChromaDB Connection Failed

**Symptoms:** `Cannot connect to ChromaDB`

**Solutions:**
```bash
# Check logs
docker logs chromadb_phd

# Restart service
docker-compose restart chromadb

# Verify heartbeat
curl http://localhost:8000/api/v2/heartbeat
```

### Memory Issues

**Symptoms:** Container killed, OOM errors

**Solutions:**
```bash
# Reduce batch size
docker-compose run --rm rag_uq python experiments/run_router_training.py \
    --batch-size 8 --synthetic

# Increase Docker memory in Docker Desktop settings
```

### Port Already in Use

**Symptoms:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solutions:**
```bash
# Find process using port
lsof -i :8000

# Kill the process or change port in docker-compose.yml
```

### Model Download Fails

**Symptoms:** Timeout or network errors pulling models

**Solutions:**
```bash
# Manual pull with retry
docker exec -it ollama_phd ollama pull llama3.2:3b

# Check Ollama storage
docker exec ollama_phd du -sh /root/.ollama
```

## Running Tests

```bash
docker-compose run --rm rag_uq pytest tests/ -v
```

## Getting Help

1. Check container logs: `docker-compose logs`
2. Verify services: `docker ps`
3. Open an issue on GitHub

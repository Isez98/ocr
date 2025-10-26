#!/bin/bash
# Migration script from old structure to new modular structure

echo "🔄 Migrating OCR project to modular structure..."

# Create backup of current app.py
if [ -f "app.py" ]; then
    echo "📁 Backing up original app.py..."
    cp app.py app.py.backup
    echo "✅ Backup created: app.py.backup"
fi

# Update Dockerfile to use new entry point
echo "🐳 Updating Dockerfile..."
if [ -f "Dockerfile" ]; then
    sed -i 's/CMD \["uvicorn", "app:app"/CMD ["python", "main.py"]/g' Dockerfile
    echo "✅ Dockerfile updated"
fi

# Create startup script for development
cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting OCR Service (Development Mode)"
python main.py
EOF

chmod +x start_dev.sh

# Create production startup script
cat > start_prod.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting OCR Service (Production Mode)"
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --workers 4
EOF

chmod +x start_prod.sh

echo ""
echo "✅ Migration completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Test the new API: python main.py"
echo "2. Verify endpoints: curl http://localhost:8001/health"
echo "3. Check processing: curl -X POST 'http://localhost:8001/api/v1/process' -F 'template_id=form_v1' -F 'file=@sample.jpg'"
echo ""
echo "🔧 Development commands:"
echo "- Start dev server: ./start_dev.sh"
echo "- Start prod server: ./start_prod.sh"
echo "- Run tests: python -m pytest tests/"
echo ""
echo "📁 New structure:"
echo "- API endpoints: src/api/endpoints/"
echo "- Core logic: src/core/"
echo "- ML components: src/ml/"
echo "- Utilities: src/utils/"
echo "- Tools: tools/"
echo "- Tests: tests/"
echo "- Data: data/"
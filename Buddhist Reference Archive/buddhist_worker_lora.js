// buddhist-ai-training/worker/src/index.js
// Updated worker with LoRA integration

/**
 * Buddhist AI Worker with LoRA Fine-Tuning
 * 
 * This worker provides Buddhist ethical guidance using a fine-tuned
 * LoRA adapter trained on Buddhist principles.
 */

export default {
  async fetch(request, env, ctx) {
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // Health check endpoint
    if (request.url.endsWith('/health')) {
      return Response.json({ 
        status: 'healthy',
        lora_version: env.LORA_VERSION || 'buddhist-ethics-lora-v1'
      }, { headers: corsHeaders });
    }

    // Main endpoint
    if (request.method === 'POST') {
      try {
        const { scenario, context, history } = await request.json();

        if (!scenario) {
          return Response.json(
            { error: 'Scenario is required' },
            { status: 400, headers: corsHeaders }
          );
        }

        // Build conversation history if provided
        const messages = [];
        
        // System message with Buddhist context
        messages.push({
          role: 'system',
          content: `You are an AI assistant trained in Buddhist ethics, philosophy, and practice. Your guidance is rooted in:

- The Four Noble Truths (understanding suffering and its cessation)
- The Noble Eightfold Path (the path to liberation)
- The Five Precepts (ethical conduct)
- The Brahma Viharas (loving-kindness, compassion, empathetic joy, equanimity)
- Dependent Origination (interconnectedness of all phenomena)

Provide compassionate, wise, and practical guidance that honors the questioner's experience while offering Buddhist perspective.`
        });

        // Add conversation history if provided
        if (history && Array.isArray(history)) {
          messages.push(...history);
        }

        // Add current scenario
        messages.push({
          role: 'user',
          content: scenario
        });

        // Call Workers AI with LoRA adapter
        const startTime = Date.now();
        
        const aiResponse = await env.AI.run(
          '@cf/mistral/mistral-7b-instruct-v0.2-lora',
          {
            messages: messages,
            lora: env.LORA_VERSION || 'buddhist-ethics-lora-v1',
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9
          }
        );

        const inferenceTime = Date.now() - startTime;

        // Extract response
        const guidance = aiResponse.response || aiResponse.result?.response;

        // Log to database for RL feedback and analysis
        if (env.DB) {
          try {
            await env.DB.prepare(`
              INSERT INTO ethical_responses 
              (scenario, response, inference_time, lora_version, timestamp)
              VALUES (?, ?, ?, ?, ?)
            `).bind(
              scenario,
              guidance,
              inferenceTime,
              env.LORA_VERSION || 'buddhist-ethics-lora-v1',
              new Date().toISOString()
            ).run();
          } catch (dbError) {
            console.error('Database logging error:', dbError);
            // Continue even if logging fails
          }
        }

        // Return response
        return Response.json({
          guidance,
          inference_time_ms: inferenceTime,
          lora_version: env.LORA_VERSION || 'buddhist-ethics-lora-v1',
          timestamp: new Date().toISOString()
        }, { headers: corsHeaders });

      } catch (error) {
        console.error('Error processing request:', error);
        
        return Response.json({
          error: 'Failed to generate guidance',
          message: error.message
        }, { 
          status: 500,
          headers: corsHeaders 
        });
      }
    }

    // Get recent interactions (for analysis)
    if (request.method === 'GET' && request.url.endsWith('/recent')) {
      if (!env.DB) {
        return Response.json(
          { error: 'Database not configured' },
          { status: 500, headers: corsHeaders }
        );
      }

      try {
        const results = await env.DB.prepare(`
          SELECT scenario, response, inference_time, timestamp
          FROM ethical_responses
          ORDER BY timestamp DESC
          LIMIT 10
        `).all();

        return Response.json({
          interactions: results.results,
          count: results.results.length
        }, { headers: corsHeaders });

      } catch (error) {
        console.error('Database query error:', error);
        
        return Response.json({
          error: 'Failed to fetch recent interactions',
          message: error.message
        }, { 
          status: 500,
          headers: corsHeaders 
        });
      }
    }

    // Default response
    return Response.json({
      error: 'Invalid endpoint',
      available_endpoints: [
        'POST / - Get Buddhist ethical guidance',
        'GET /health - Health check',
        'GET /recent - View recent interactions'
      ]
    }, { 
      status: 404,
      headers: corsHeaders 
    });
  }
};

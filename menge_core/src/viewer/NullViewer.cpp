/*

License

Menge
Copyright � and trademark � 2012-14 University of North Carolina at Chapel Hill. 
All rights reserved.

Permission to use, copy, modify, and distribute this software and its documentation 
for educational, research, and non-profit purposes, without fee, and without a 
written agreement is hereby granted, provided that the above copyright notice, 
this paragraph, and the following four paragraphs appear in all copies.

This software program and documentation are copyrighted by the University of North 
Carolina at Chapel Hill. The software program and documentation are supplied "as is," 
without any accompanying services from the University of North Carolina at Chapel 
Hill or the authors. The University of North Carolina at Chapel Hill and the 
authors do not warrant that the operation of the program will be uninterrupted 
or error-free. The end-user understands that the program was developed for research 
purposes and is advised not to rely exclusively on the program for any reason.

IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR THE AUTHORS 
BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL 
DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS 
DOCUMENTATION, EVEN IF THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR THE 
AUTHORS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND THE AUTHORS SPECIFICALLY 
DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND ANY STATUTORY WARRANTY 
OF NON-INFRINGEMENT. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND 
THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND THE AUTHORS HAVE NO OBLIGATIONS 
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Any questions or comments should be sent to the authors {menge,geom}@cs.unc.edu

*/

#include "NullViewer.h"
#include "GLScene.h"

namespace Menge {

	namespace Vis {

		/////////////////////////////////////////////////////////////////////////////
		//                     Implementation of NullViewer
		/////////////////////////////////////////////////////////////////////////////

		NullViewer::NullViewer(): _scene(0x0), _stepSize(0.1f), _requestedSteps(0) {
		}

		/////////////////////////////////////////////////////////////////////////////

		NullViewer::~NullViewer() {
			if ( _scene ) {
				delete _scene;
			}
		}

		/////////////////////////////////////////////////////////////////////////////

		void NullViewer::setScene( SceneGraph::GLScene * scene ) {
			if ( _scene ) {
				delete _scene;
			}
			_scene = scene;
		}

		///////////////////////////////////////////////////////////////////////////

		void NullViewer::setFixedStep( float stepSize ) {
			_stepSize = stepSize;
		}

        ///////////////////////////////////////////////////////////////////////////

        void NullViewer::setStepFromMsg(const std_msgs::UInt8::ConstPtr& msg) {
            //copy from message and into step variables
            ROS_DEBUG("msg received on /step:\n\tnumber of sim steps to take : [%d]", msg->data);
            _step = true;
            _requestedSteps = msg->data;
        }

        void NullViewer::setRunFromMsg(const std_msgs::Bool::ConstPtr& msg) {
            //copy from message and into pause
            ROS_DEBUG("msg received on /run:\n\tSimulation set to running : [%s]", msg->data ? "true" : "false");
            _pause = !msg->data;
        }

		/////////////////////////////////////////////////////////////////////////////

		void NullViewer::run(ros::CallbackQueue &queue) {

			if ( _scene == 0x0 ) return;
			_viewTime = 0.f;
			float frameLen = 0.f;
            _scene_updated = false;
            _pause = true;
            bool lastItrPaused = _pause;
            _step = false;
            _requestedSteps = 0;
            uint8_t stepCounter = 0;
			_fpsTimer.start();
            std_msgs::Float32 time_msg;

			while ( ros::ok() ) {
                queue.callAvailable(ros::WallDuration());

                // restart timer after pause
                if (lastItrPaused && !_pause) {
                    ROS_DEBUG("Switched from pause to running after update");
//                    ros::getGlobalCallbackQueue()->clear();
                    _fpsTimer.restart();
                }

                if (_scene_updated) {
                    _scene_updated = false;
                }
                // make simulation step if requested
                if (_pause && _step) {
                    if (stepCounter == 0) {
                        ROS_DEBUG("Perform [%d] simulation steps", _requestedSteps);
                        ros::getGlobalCallbackQueue()->clear();
                    }
                    stepCounter += 1;

                    if (stepCounter > _requestedSteps) {
                        ROS_DEBUG("Performed requested steps");
                        _step = false;
                        stepCounter = 0;
                        _requestedSteps = 0;

                    } else {
                        _viewTime += _stepSize;
                        _scene_updated = true;
                    }
                } else if (!_pause) {
                    _viewTime += _stepSize;
                    _scene_updated = true;
                }

                if (_scene_updated) {
                    ROS_INFO("Advance simulation");
                    try {
                        _scene->updateScene(_viewTime);
                        _fpsTimer.lap();
                        time_msg.data = _viewTime;
                        _pub_time.publish(time_msg);
                    } catch (SceneGraph::SystemStopException) {
                        break;
                    }
                }

                lastItrPaused = _pause;
            }
			std::cout << "Average frame computation time: " << _fpsTimer.average( 0.001f ) << " ms\n";
			_scene->finish();

		}
	}	 // namespace Vis
}	// namespace Menge